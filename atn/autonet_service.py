"""Autonet network service integration for ATN.

Bridges the ATN Runtime with the Autonet decentralized training service.
The service is optional — everything works without it.  When enabled
(via config or user toggle), it starts the training loop in the background.

Lifecycle:
  - ATN Runtime starts → AutonetBridge.start() (if enabled in config)
  - User toggles training on → AutonetBridge.start()
  - User toggles training off → AutonetBridge.stop()
  - ATN Runtime stops → AutonetBridge.stop()

The bridge exposes status and control methods consumed by WS handlers.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

from .events import EventType

if TYPE_CHECKING:
    from .config import AutonetConfig
    from .events import EventBus

log = logging.getLogger(__name__)


class AutonetStatus(str, Enum):
    """Status of the autonet network service."""
    DISABLED = "disabled"       # Not configured / user hasn't opted in
    STOPPED = "stopped"         # Configured but not running
    STARTING = "starting"       # Starting up
    RUNNING = "running"         # Training loop active
    PAUSED = "paused"           # Paused (resource limits, user pause)
    STOPPING = "stopping"       # Shutting down
    ERROR = "error"             # Failed to start or crashed


def _charter_axes() -> list[dict[str, str]]:
    try:
        from nodes.common.library.charter import active_charter
        return [{"id": i, "thesis": t} for i, t in active_charter().axes]
    except Exception:
        return []


def _stable_p2p_port(data_dir, host: str = "0.0.0.0") -> int:
    """The p2p port saved in ``<data_dir>/p2p.port``; on first use a free
    port is picked and saved. 0 (random) when neither works."""
    import socket
    from pathlib import Path as _Path
    path = _Path(data_dir) / "p2p.port"
    try:
        port = int(path.read_text(encoding="utf-8").strip())
        if 0 < port < 65536:
            return port
    except Exception:  # noqa: BLE001 (missing or unreadable: pick one)
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = s.getsockname()[1]
        path.write_text(str(port), encoding="utf-8")
        return port
    except Exception:  # noqa: BLE001
        log.warning("could not save a p2p port in %s; using a random one", path)
        return 0


@dataclass
class AutonetState:
    """Observable state of the autonet service."""
    status: AutonetStatus = AutonetStatus.DISABLED
    wallet_connected: bool = False
    wallet_address: str = ""
    chain_id: int = 0
    rpc_url: str = ""
    jurisdiction_name: str = ""
    # Training metrics
    cycles_completed: int = 0
    uptime_seconds: float = 0.0
    errors: int = 0
    last_error: str = ""
    paused_reason: str = ""
    # Per-cycle training metrics (from TrainingDataFeed)
    training_loss: float = 0.0
    training_batches: int = 0
    training_cosine_sim: float = 0.0
    training_segments: int = 0
    pending_events: int = 0
    last_cycle_time: float = 0.0
    # Training history (last N cycles for charting)
    loss_history: list[float] = field(default_factory=list)
    # Behavioral profile
    profile_hash: str = ""
    # Resource usage: no longer sampled (the CPU gate and JEPA training are
    # retired); kept at zero for the app's status shape.
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_available: bool = False
    gpu_memory_mb: float = 0.0
    # Contract addresses (discovered from chain)
    dao_address: str = ""
    rpb_contract_address: str = ""    # Legacy: pre-substrate RPB; empty post-redeploy
    substrate_address: str = ""        # Phase 12: deployed Substrate.sol
    # Native gas token metadata (mirrored from config so the UI can label
    # balances without hardcoding 'ETH').
    gas_symbol: str = "gas"
    gas_decimals: int = 18
    # The jurisdiction's token symbols, read from its RepToken and
    # Substrate; the app labels with these and falls back to REP / ATN.
    rep_symbol: str = ""
    atn_symbol: str = ""
    registry_address: str = ""
    token_address: str = ""
    economy_address: str = ""
    timelock_address: str = ""
    # Local charter vs the jurisdiction's governed charter (registry key
    # jurisdiction.<id>.charter): True / False, None = none published.
    charter_match: bool | None = None
    # The jurisdiction's identity "<chain_id>:<DAO address>" (gossip topic,
    # substrate data key) and why it can't start, if it can't.
    jurisdiction_key: str = ""
    dao_registry_address: str = ""
    tool_bond_address: str = ""
    epoch_rules: str = ""
    jurisdiction_errors: list[str] = field(default_factory=list)
    # False until a jurisdiction is chosen (issue #50).
    jurisdiction_configured: bool = True
    # Retired (agent prompts carry no jurisdiction text); always empty, kept
    # on the wire because the app's AutonetState still parses them.
    constitution_cid: str = ""
    constitution_text: str = ""

    _MAX_LOSS_HISTORY: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "wallet_connected": self.wallet_connected,
            "wallet_address": self.wallet_address,
            "chain_id": self.chain_id,
            "rpc_url": self.rpc_url,
            "jurisdiction_name": self.jurisdiction_name,
            "cycles_completed": self.cycles_completed,
            "uptime_seconds": self.uptime_seconds,
            "errors": self.errors,
            "last_error": self.last_error,
            "paused_reason": self.paused_reason,
            # Training metrics
            "training": {
                "loss": self.training_loss,
                "batches": self.training_batches,
                "cosine_similarity": self.training_cosine_sim,
                "segments_loaded": self.training_segments,
                "pending_events": self.pending_events,
                "last_cycle_time": self.last_cycle_time,
                "loss_history": self.loss_history,
                "profile_hash": self.profile_hash,
            },
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "gpu_available": self.gpu_available,
            "gpu_memory_mb": self.gpu_memory_mb,
            "dao_address": self.dao_address,
            "rpb_contract_address": self.rpb_contract_address,
            "substrate_address": self.substrate_address,
            "gas_symbol": self.gas_symbol,
            "gas_decimals": self.gas_decimals,
            "rep_symbol": self.rep_symbol,
            "atn_symbol": self.atn_symbol,
            "registry_address": self.registry_address,
            "token_address": self.token_address,
            "economy_address": self.economy_address,
            "timelock_address": self.timelock_address,
            "charter_match": self.charter_match,
            "jurisdiction_key": self.jurisdiction_key,
            "dao_registry_address": self.dao_registry_address,
            "tool_bond_address": self.tool_bond_address,
            "epoch_rules": self.epoch_rules,
            "jurisdiction_errors": list(self.jurisdiction_errors),
            "jurisdiction_configured": self.jurisdiction_configured,
            # The active charter's axes, in coordinate order: what the UI
            # draws (a DAO defines its own; docs/multi_jurisdiction.md 5).
            "charter_axes": _charter_axes(),
            "constitution_cid": self.constitution_cid,
            "constitution_text": self.constitution_text,
        }


class AutonetBridge:
    """Bridge between ATN Runtime and the Autonet training service.

    Manages the lifecycle of the AutonetService and exposes control/status
    methods for the WS API.  Runs the training loop in a background asyncio
    task so it doesn't block the ATN event loop.
    """

    def __init__(self, config: AutonetConfig, event_bus: EventBus | None = None,
                 data_dir: str = "") -> None:
        self.config = config
        self.state = AutonetState()
        self._events = event_bus
        self._service = None      # Will hold nodes.service.AutonetService
        self._task: asyncio.Task | None = None
        # Background drain of remote tool manifests into the discovery
        # index (network tool search). Spawned once gossip is wired.
        self._manifest_pull_task: asyncio.Task | None = None
        self._service_catalog_task: asyncio.Task | None = None
        # Staked review driver (atn/review_driver.py): commits, reveals,
        # finalizes and claims this daemon's agents' ToolBond review
        # batches. Runs while the bridge runs.
        self._review_driver: Any = None
        self._review_driver_task: asyncio.Task | None = None
        self._autonet_config = None  # Will hold nodes.common.config.AutonetConfig
        # Training data feed (Story 3.2) — ATN data dir for JSONL files
        self._data_dir = data_dir
        # The jurisdiction's sanctions in force (atn/sanctions.py), built
        # on first use from the resolved chain addresses.
        self._sanctions: Any = None
        self._sanctions_key: tuple[str, str, str] | None = None

        # Apply config values to state
        if config.wallet_address:
            self.state.wallet_connected = True
            self.state.wallet_address = config.wallet_address
        if config.rpc_url:
            self.state.rpc_url = config.rpc_url
        if config.chain_id:
            self.state.chain_id = config.chain_id

        # Surface whatever the config already holds. A jurisdiction the owner
        # chose (issue #50) is read at boot: its Registry names the contracts
        # the app shows and registration needs. Without a choice nothing is
        # read until something joins.
        self._mirror_jurisdiction()
        from .config import _selection_data_dir, load_selected_jurisdiction
        if load_selected_jurisdiction(_selection_data_dir()) is not None:
            self.ensure_network_config()

        # Runtime back-reference (set by Runtime.__init__)
        self._runtime = None
        # Main event loop, captured at start (charter-change restarts are
        # requested from the nodes worker thread).
        self._loop: asyncio.AbstractEventLoop | None = None

        # P2P agent advertisement
        self._p2p_host = None      # AutonetHost (lazy, trio-based)
        # The daemon's asyncio service_request handler and its loop, set by
        # the WS bridge, so libp2p service calls reach the same code path as
        # WS ones (installed on every host start_p2p builds).
        self._service_request_handler = None
        self._service_request_loop: asyncio.AbstractEventLoop | None = None
        self._p2p_thread: threading.Thread | None = None
        self._p2p_stop = threading.Event()
        self._agent_registry = None  # Set by Runtime after init
        # Subscribe to agent lifecycle events for p2p advertisement refresh
        if self._events:
            for evt in (EventType.AGENT_REGISTERED, EventType.AGENT_UNREGISTERED,
                        EventType.AGENT_ACTIVATED, EventType.AGENT_DEACTIVATED):
                self._events.subscribe(evt, self._on_agent_changed)

    @property
    def sanctions(self) -> Any:
        """The jurisdiction's sanction list (atn/sanctions.py), rebuilt when
        the chain addresses change (a jurisdiction switch restarts anyway)."""
        from .sanctions import SanctionList
        c = self.config
        key = (str(getattr(c, "rpc_url", "") or ""),
               str(getattr(c, "dao_registry_address", "") or ""),
               str(getattr(c, "substrate_address", "") or ""))
        if self._sanctions is None or self._sanctions_key != key:
            self._sanctions = SanctionList.from_config(c)
            self._sanctions_key = key
        return self._sanctions

    @property
    def network_joined(self) -> bool:
        """True once the daemon joined the network (first on-chain
        registration, or ``autonet.enabled``). Before that it is fully local
        and read-only chain calls answer "not joined"."""
        return bool(getattr(self.config, "network_joined", False))

    def ensure_network_config(self) -> bool:
        """Join the network: resolve the packaged registry (or the operator's
        ATN_REGISTRY_URL) into the config and refresh the surfaced state.
        Idempotent. Called by start() and by the registration paths before
        they touch the chain. Returns ``network_joined``."""
        if self.network_joined:
            return True
        try:
            from .config import resolve_network_registry
            resolve_network_registry(self.config)
        except Exception:
            log.warning("Network registry resolution failed", exc_info=True)
            return False
        if self.config.rpc_url:
            self.state.rpc_url = self.config.rpc_url
        if self.config.chain_id:
            self.state.chain_id = self.config.chain_id
        self._mirror_jurisdiction()
        return self.network_joined

    def _mirror_jurisdiction(self) -> None:
        """Mirror the resolved jurisdiction into the WS state and record why
        it can't start, if it can't (docs/multi_jurisdiction.md decisions
        3-5). Every address comes from one resolution of the chosen DAO's
        Registry (or the packaged registry.json that names it)."""
        from .config import (epoch_rules_error, gas_symbol_for_chain,
                             jurisdiction_key, missing_prerequisites)

        c = self.config
        st = self.state
        st.gas_symbol = (getattr(c, "gas_symbol", "") or
                         gas_symbol_for_chain(getattr(c, "chain_id", 0)))
        st.gas_decimals = int(getattr(c, "gas_decimals", 18) or 18)
        st.dao_address = c.dao_address
        st.rpb_contract_address = c.rpb_contract_address
        st.substrate_address = getattr(c, "substrate_address", "")
        st.registry_address = c.registry_address
        st.dao_registry_address = getattr(c, "dao_registry_address", "")
        st.tool_bond_address = getattr(c, "tool_bond_address", "")
        st.token_address = c.token_address
        st.economy_address = c.economy_address
        st.timelock_address = c.timelock_address
        from .config import CURRENT_EPOCH_RULES
        st.epoch_rules = (getattr(c, "epoch_rules", "") or CURRENT_EPOCH_RULES)
        st.jurisdiction_key = jurisdiction_key(c)
        st.rpc_url = c.rpc_url or ""
        st.chain_id = int(c.chain_id or 0)
        # A jurisdiction is called by its DAO's name: the saved choice's,
        # else the governor's name() once joined.
        from .config import _selection_data_dir, dao_name, load_selected_jurisdiction
        sel = load_selected_jurisdiction(_selection_data_dir())
        st.jurisdiction_name = ((sel or {}).get("name") or
                                (dao_name(c.dao_address, c.rpc_url)
                                 if self.network_joined else ""))
        from .config import token_symbol
        st.rep_symbol = token_symbol(getattr(c, "rep_token_address", ""), c.rpc_url)
        st.atn_symbol = token_symbol(st.substrate_address, c.rpc_url)

        errors: list[str] = []
        from .config import NO_JURISDICTION_MESSAGE, jurisdiction_configured
        # Addresses set in config.yaml configure a jurisdiction too.
        st.jurisdiction_configured = (jurisdiction_configured()
                                      or bool(getattr(c, "substrate_address", "")))
        if self.network_joined and not st.jurisdiction_configured:
            errors.append(NO_JURISDICTION_MESSAGE)
        elif self.network_joined:
            conflict = getattr(c, "registry_conflict", "")
            if conflict:
                errors.append(conflict)
            missing = missing_prerequisites(c)
            if missing:
                errors.append("missing jurisdiction addresses: " + ", ".join(missing))
            rules = epoch_rules_error(c)
            if rules:
                errors.append(rules)
            charter = self._charter_error()
            if charter:
                errors.append(charter)
        st.jurisdiction_errors = errors
        for e in errors:
            log.warning("Jurisdiction %s can't start: %s",
                        st.jurisdiction_key or "(unknown)", e)

    def _charter_error(self) -> str:
        """Check the jurisdiction's charter text is one this build can run.

        The Registry holds the charter text (2026-10-17) and each DAO
        defines its own axes; only text ``Charter.from_text`` rejects is
        refused. ``start`` installs a valid charter as the active one. A
        missing charter (pre-2026-10-17 deploys) falls back to the hash
        comparison, which only warns."""
        text = getattr(self.config, "charter_text", "") or ""
        if not text:
            try:
                from .on_chain import verify_charter
                self.state.charter_match = verify_charter(
                    getattr(self.config, "charter_hash", ""),
                    getattr(self.config, "charter_uri", ""),
                )["match"]
            except Exception:
                log.debug("charter check skipped", exc_info=True)
            return ""
        from .config import charter_text_error
        _is_default, error = charter_text_error(text)
        # A valid charter is installed and run, so local == jurisdiction.
        self.state.charter_match = None if error else True
        return error

    def _install_charter(self) -> None:
        """Make the jurisdiction's charter the process's active charter.

        Called at join, BEFORE the nodes-side service builds any world:
        every charter root, coordinate head and the close's axis set come
        from it. No charter text (pre-2026-10-17 deploys) keeps the
        built-in DEFAULT_CHARTER."""
        from nodes.common.library.charter import (
            DEFAULT_CHARTER,
            Charter,
            set_active_charter,
        )
        text = getattr(self.config, "charter_text", "") or ""
        charter = Charter.from_text(text) if text else DEFAULT_CHARTER
        set_active_charter(charter)
        log.info("Active charter: %d axes (%s), hash=%s",
                 charter.n_dims, ",".join(charter.ids), charter.hash)

    def _on_charter_changed(self, new_text: str) -> None:
        """The close driver saw the jurisdiction's charter change (pinned
        to the previous anchor). Every daemon sees the same pinned value at
        the same epoch, so each restarts onto the new charter. Called from
        the nodes worker thread."""
        msg = "the jurisdiction changed its charter; restarting"
        log.warning("%s", msg)
        self.state.jurisdiction_errors = list(
            dict.fromkeys(list(self.state.jurisdiction_errors) + [msg]))
        self.state.last_error = msg

        def _restart() -> None:
            try:
                from . import cli as _cli_module
                _cli_module._restart_requested = True
                if self._runtime is not None:
                    self._runtime._shutdown_event.set()
            except Exception:
                log.exception("charter-change restart request failed")

        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(_restart)
        else:
            _restart()

    # ------------------------------------------------------------------
    # P2P agent advertisement
    # ------------------------------------------------------------------

    def set_agent_registry(self, registry: Any) -> None:
        """Called by Runtime after init to provide agent registry reference."""
        self._agent_registry = registry

    async def _on_agent_changed(self, event: Any) -> None:
        """Refresh p2p capability when agents are registered/unregistered."""
        self._refresh_p2p_agents()

    def _refresh_p2p_agents(self) -> None:
        """Rebuild agent advertisements from registry and push to p2p host."""
        if not self._p2p_host or not self._agent_registry:
            return
        try:
            ads = self._agent_registry.build_agent_advertisements()
            # When this daemon is a sponsor (work-AI employer), mark its
            # agents as sponsors so dependents on other daemons can discover
            # them and target by address. Over-advertising is safe: the
            # handler still authorizes per-dependent against the binding
            # store, so an unbound requester is rejected regardless.
            if getattr(self.config, "sponsor_inference", False):
                sponsor_model = getattr(self.config, "sponsor_model", "") or ""
                for ad in ads:
                    ad["is_sponsor"] = True
                    if sponsor_model and not ad.get("model"):
                        ad["model"] = sponsor_model
            # Also include the connected wallet as a root agent if present
            if self.state.wallet_address and not any(
                a["address"].lower() == self.state.wallet_address.lower() for a in ads
            ):
                # self.config IS the RPBConfig (sponsor_* fields live directly
                # on it — there is no nested .rpb attribute).
                rpb_cfg = self.config
                ads.insert(0, {
                    "address": self.state.wallet_address,
                    "name": "root",
                    "description": "",
                    "agent_type": "root",
                    "model": getattr(rpb_cfg, "sponsor_model", "") or "",
                    "is_root": True,
                    "parent_address": "",
                    "registered_on_chain": False,
                    "is_sponsor": getattr(rpb_cfg, "sponsor_inference", False),
                })
            self._p2p_host.update_capability(agents=ads)
            log.debug("P2P capability updated with %d agent(s)", len(ads))
        except Exception:
            log.debug("Failed to refresh p2p agents", exc_info=True)

    def start_p2p(self) -> None:
        """Start the p2p host in a background thread for agent advertisement."""
        if self._p2p_thread and self._p2p_thread.is_alive():
            return
        try:
            from nodes.common.p2p import AutonetHost, NodeCapability
            from nodes.common.config import load_config as load_autonet_config
        except Exception:
            log.debug("P2P not available (nodes package not installed or import error)")
            return

        config_path = self.config.config_path or None
        try:
            cfg = load_autonet_config(config_path)
        except Exception:
            cfg = None

        listen_port = cfg.p2p.listen_port if cfg else 0
        listen_host = cfg.p2p.listen_host if cfg else "0.0.0.0"
        if not listen_port and self._data_dir:
            # Keep one port across restarts: the daemon's address can be a
            # Registry seed, and a new random port each start made it stale.
            listen_port = _stable_p2p_port(self._data_dir, listen_host)
        # Phase 12: a fresh install with no user-configured bootstrap
        # falls back to community-published bootstrap peers, so daemons
        # can join without manual multiaddr configuration.
        from nodes.common.peer_attribution import resolve_bootstrap_peers
        # Local config wins; else the jurisdiction's Registry seeds.
        bootstrap = resolve_bootstrap_peers(
            (cfg.p2p.bootstrap_peers if cfg else [])
            or list(getattr(self.config, "bootstrap_peers", []) or []))
        advertise_interval = cfg.p2p.capability_advertise_interval if cfg else 60
        # Registry seeds are re-read while running (#42); seeds set in the
        # local config are fixed.
        seed_source = None
        if not (cfg and cfg.p2p.bootstrap_peers):
            from .config import registry_seed_source
            seed_source = registry_seed_source(self.config)

        node_id = f"atn-{self.state.wallet_address[:8]}" if self.state.wallet_address else "atn-daemon"
        cap = NodeCapability(peer_id="", node_id=node_id)

        # A stable libp2p identity: agents register this peer id on chain,
        # so a new one per restart left their on-chain peer id stale.
        key_seed = None
        if self._data_dir:
            from pathlib import Path as _Path
            from nodes.relay import load_key_seed
            try:
                key_seed = load_key_seed(_Path(self._data_dir) / "p2p.key")
            except Exception as exc:  # noqa: BLE001
                log.warning("p2p.key unreadable (%s); using a fresh peer id", exc)

        host = AutonetHost(
            node_id=node_id,
            key_seed=key_seed,
            listen_port=listen_port,
            listen_host=listen_host,
            bootstrap_peers=bootstrap,
            capability=cap,
            announce_addrs=cfg.p2p.announce_addrs if cfg else None,
            bootstrap_redial_interval=cfg.p2p.bootstrap_redial_interval if cfg else 30,
            enable_upnp=cfg.p2p.enable_upnp if cfg else True,
            auto_reach=cfg.p2p.auto_reach if cfg else True,
            reach_interval=cfg.p2p.reach_interval if cfg else 300,
            transport=cfg.p2p.transport if cfg else "ws",
            seed_source=seed_source,
            seed_refresh_interval=cfg.p2p.seed_refresh_interval if cfg else 300,
        )
        self._p2p_host = host
        self._p2p_stop.clear()
        self._install_service_handler(host)
        from . import service_client
        service_client.set_p2p_host(host)

        # Wire sponsor-side inference handler (Path A). self.config IS the
        # RPBConfig — sponsor_* fields are directly on it.
        rpb_cfg = self.config
        if getattr(rpb_cfg, 'sponsor_inference', False):
            host._inference_handler = self._create_sponsor_handler(rpb_cfg)
            log.info("Sponsor inference handler wired (provider=%s, model=%s)",
                     rpb_cfg.sponsor_provider or "auto", rpb_cfg.sponsor_model or "any")

        def _run():
            import trio
            async def _main():
                async with host.run():
                    self._refresh_p2p_agents()
                    await host.advertise_capability()
                    log.info("P2P host running, advertising %d agent(s)",
                             len(host._capability.agents))
                    while not self._p2p_stop.is_set():
                        await trio.sleep(advertise_interval)
                        self._refresh_p2p_agents()
                        await host.advertise_capability()
            try:
                trio.run(_main)
            except Exception:
                log.debug("P2P host stopped", exc_info=True)

        self._p2p_thread = threading.Thread(target=_run, name="p2p-host", daemon=True)
        self._p2p_thread.start()
        log.info("P2P agent advertisement started")

    def _create_sponsor_handler(self, rpb_cfg):
        """Create an async inference handler for sponsor-side Path A.

        When a dependent node sends an inference request over P2P, this
        handler forwards it through the sponsor's own centralized provider
        (Anthropic, OpenAI, etc.) and returns the response.
        """
        from .providers.base import ToolDefinition

        service = self  # closure reference

        async def _handle_sponsor_inference(request: dict) -> dict:
            # Authorization: the sponsor is the resource owner and only serves
            # dependents it has explicitly bound (work-AI). The dependent's
            # on-chain address travels in the request body. Everything the
            # dependent spawns presents as this same address and draws on the
            # same budget — one single thread per dependent, no tree.
            dependent = request.get("agent_address", "")
            if not dependent:
                return {"error": "Missing agent_address — sponsor only serves bound dependents"}

            bindings = getattr(service._runtime, "sponsor_bindings", None) if service._runtime else None
            if bindings is None:
                return {"error": "Sponsor has no binding store"}

            binding = bindings.get(dependent)
            if binding is None:
                return {"error": "not an authorized dependent"}
            if not binding.unlimited and binding.remaining() <= 0:
                return {"error": "budget exhausted"}

            # Note: the via_rpb no-chaining guard is intentionally NOT applied
            # here. Under the single-thread model this is a bound dependent
            # consuming its own grant, not a relay chain. Only bound addresses
            # are served at all, which is the real safety boundary.

            # The employer chooses the tool (docs/sponsored_inference.md):
            # a configured sponsor_model always wins. The dependent's request
            # is only honoured when the sponsor left the model blank. Reading
            # the request first let a dependent bill the sponsor's key for any
            # model that key could reach, since the dependent always sets it.
            model = (rpb_cfg.sponsor_model or "") or request.get("model", "")
            messages = request.get("messages", [])
            system = request.get("system", "")
            try:
                max_tokens = int(request.get("max_tokens", 4096))
            except (TypeError, ValueError):
                max_tokens = 4096
            max_tokens = max(1, max_tokens)
            temperature = request.get("temperature", 0.0)
            tools_raw = request.get("tools", [])

            # Pre-flight clamp. The gate above only asks whether ANYTHING is
            # left, so a dependent with 1 token remaining could still ask for
            # a 64k completion and the sponsor would pay for all of it before
            # the next request is refused. Cap the ceiling at what is actually
            # granted, and refuse outright when the prompt alone already
            # overruns the grant (rough 4-chars-per-token estimate).
            # Concurrent requests share what is left: each reserves its worst
            # case (prompt estimate + ceiling) before the call and releases it
            # after, so two in flight can't each be clamped to all of it.
            reservation = 0
            if not binding.unlimited:
                rem = bindings.available(dependent)
                prompt_tokens = (sum(len(str(m)) for m in messages)
                                 + len(str(system))) // 4
                if prompt_tokens >= rem:
                    return {"error": "request exceeds remaining grant"
                            if prompt_tokens >= binding.remaining()
                            else "budget exhausted (held by requests in flight)"}
                max_tokens = max(1, min(max_tokens, rem - prompt_tokens))
                reservation = prompt_tokens + max_tokens
                if not bindings.reserve(dependent, reservation):
                    return {"error": "budget exhausted (held by requests in flight)"}

            # Resolve provider — use configured sponsor provider or auto-resolve
            provider = service._resolve_sponsor_provider(rpb_cfg, model)
            if provider is None:
                bindings.release(dependent, reservation)
                return {"error": "Sponsor node has no provider configured for this model"}

            # Convert tool dicts to ToolDefinition
            tools = None
            if tools_raw:
                tools = [
                    ToolDefinition(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("input_schema", {}),
                    )
                    for t in tools_raw
                ]

            try:
                resp = await provider.send(
                    messages=messages,
                    system=system,
                    model=model,
                    max_tokens=max_tokens,
                    tools=tools,
                    temperature=temperature,
                )

                # Meter the dependent's grant: spend = input + output tokens.
                in_tok = resp.usage.input_tokens if resp.usage else 0
                out_tok = resp.usage.output_tokens if resp.usage else 0
                bindings.record_spend(dependent, in_tok + out_tok)

                # Serialize ProviderResponse back to dict for P2P transport
                result = {
                    "text": resp.text or "",
                    "model": resp.model or model,
                    "stop_reason": resp.stop_reason or "end_turn",
                    "usage": {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                    },
                    # -1 signals unlimited; lets the dependent warn/stop as the
                    # grant runs low without exposing other dependents' budgets.
                    "remaining_budget_tokens": bindings.remaining(dependent),
                }
                if resp.tool_calls:
                    result["tool_calls"] = [
                        {"id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in resp.tool_calls
                    ]

                log.debug("Sponsor served inference: model=%s, tokens=%d+%d",
                          model,
                          result["usage"]["input_tokens"],
                          result["usage"]["output_tokens"])
                return result

            except Exception as e:
                log.warning("Sponsor inference failed: %s", e)
                return {"error": f"Sponsor inference failed: {e}"}
            finally:
                bindings.release(dependent, reservation)

        return _handle_sponsor_inference

    def _resolve_sponsor_provider(self, rpb_cfg, model: str):
        """Resolve a provider instance for sponsor inference."""
        provider_name = rpb_cfg.sponsor_provider

        if not provider_name:
            # Auto-resolve from model name
            model_lower = (model or "").lower()
            if model_lower.startswith("claude"):
                provider_name = "anthropic"
            elif model_lower.startswith(("gpt", "o1", "o3", "o4")):
                provider_name = "openai"
            elif model_lower.startswith("gemini"):
                provider_name = "gemini"
            else:
                provider_name = "anthropic"  # default

        # Use the runtime's provider manager if available
        if self._runtime and hasattr(self._runtime, 'providers'):
            try:
                from .models import AgentDefinition
                dummy_defn = AgentDefinition(
                    id="_sponsor",
                    name="sponsor",
                    provider=provider_name,
                    cognitive_model=model,
                )
                return self._runtime.providers.resolve_provider_with_fallback(dummy_defn)
            except Exception:
                log.debug("Failed to resolve sponsor provider via runtime", exc_info=True)

        # Fallback: direct creation from credentials
        try:
            from pathlib import Path
            from .providers.anthropic import AnthropicProvider
            from .credentials import CredentialStore
            if self._runtime is not None and hasattr(self._runtime, "credential_store"):
                creds = self._runtime.credential_store
            else:
                creds = CredentialStore(Path.home() / ".atn")
            api_key = creds.load(f"provider_{provider_name}").get("api_key", "")
            if api_key and provider_name == "anthropic":
                return AnthropicProvider(api_key=api_key, default_model=model)
            if api_key and provider_name in ("openai", "gemini"):
                from .providers.openai_compat import OpenAICompatibleProvider
                base_urls = {
                    "openai": "https://api.openai.com/v1",
                    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
                }
                return OpenAICompatibleProvider(
                    name=f"sponsor-{provider_name}",
                    base_url=base_urls.get(provider_name, ""),
                    api_key=api_key,
                    default_model=model,
                )
        except Exception:
            log.debug("Failed to create sponsor provider directly", exc_info=True)

        return None

    def enable_sponsor_inference(self, provider: str = "", model: str = "") -> dict:
        """Turn this daemon into a sponsor (work-AI employer).

        Sets the sponsor_* config flags and, if the p2p host is already
        running, wires the sponsor handler + refreshes the advertisement so
        the change takes effect without a restart. Idempotent.
        """
        self.config.sponsor_inference = True
        if provider:
            self.config.sponsor_provider = provider
        if model:
            self.config.sponsor_model = model

        # Persist: the flags are read back from config.yaml on boot
        # (config.py), so without this the mode silently lapses on restart
        # while sponsor_bindings.json keeps listing the dependents.
        try:
            from .config import save_sponsor_mode_to_config
            save_sponsor_mode_to_config(
                True, self.config.sponsor_provider, self.config.sponsor_model,
            )
        except Exception:
            log.warning("Failed to persist sponsor mode to config", exc_info=True)

        host = self._p2p_host
        if host is not None:
            host._inference_handler = self._create_sponsor_handler(self.config)
            self._refresh_p2p_agents()
            log.info("Sponsor inference enabled at runtime (provider=%s, model=%s)",
                     self.config.sponsor_provider or "auto",
                     self.config.sponsor_model or "any")
        else:
            log.info("Sponsor inference enabled; handler wires when p2p starts")
        return {
            "sponsor_inference": True,
            "sponsor_provider": self.config.sponsor_provider,
            "sponsor_model": self.config.sponsor_model,
            "p2p_running": host is not None,
        }

    def disable_sponsor_inference(self) -> dict:
        """Stop being a sponsor: unwire the handler and drop the advert.

        The mirror of enable_sponsor_inference. Removing the last binding is
        not an off switch (the handler stays wired and ``is_sponsor`` stays on
        the advertisement), so the owner needs an explicit one. Existing
        bindings are left on disk: this turns serving off, it does not revoke
        grants. Idempotent.
        """
        self.config.sponsor_inference = False

        try:
            from .config import save_sponsor_mode_to_config
            save_sponsor_mode_to_config(
                False, self.config.sponsor_provider, self.config.sponsor_model,
            )
        except Exception:
            log.warning("Failed to persist sponsor mode to config", exc_info=True)

        host = self._p2p_host
        if host is not None:
            host._inference_handler = None
            self._refresh_p2p_agents()
            log.info("Sponsor inference disabled at runtime")
        else:
            log.info("Sponsor inference disabled; handler stays unwired when p2p starts")
        return {
            "sponsor_inference": False,
            "sponsor_provider": self.config.sponsor_provider,
            "sponsor_model": self.config.sponsor_model,
            "p2p_running": host is not None,
        }

    def stop_p2p(self) -> None:
        """Stop the p2p background thread."""
        self._p2p_stop.set()
        if self._p2p_thread:
            self._p2p_thread.join(timeout=5)
            self._p2p_thread = None
        if self._p2p_host is not None:
            from . import service_client
            service_client.clear_p2p_host(self._p2p_host)
        self._p2p_host = None

    def set_service_request_handler(self, serve, loop) -> None:
        """Serve ``/autonet/service/1.0.0`` requests with ``serve(msg) ->
        reply`` (an asyncio coroutine function run on ``loop``): the WS
        bridge's shared ``service_request`` path. Installed on the current
        host now and on every host built later."""
        self._service_request_handler = serve
        self._service_request_loop = loop
        if self._p2p_host is not None:
            self._install_service_handler(self._p2p_host)

    def _install_service_handler(self, host) -> None:
        serve, loop = self._service_request_handler, self._service_request_loop
        if serve is None or loop is None:
            return
        from .service_client import host_service_handler
        host.set_service_handler(host_service_handler(serve, loop))
        # Service streams (/autonet/service-stream/1.0.0) pass the same gate.
        from .service_streams import stream_handler_for
        host.set_service_stream_handler(stream_handler_for(serve, loop))

    async def _emit(self, event_type_name: str, data: dict[str, Any] | None = None) -> None:
        """Emit an event if the event bus is available."""
        if not self._events:
            return
        from .events import Event, EventType
        et = getattr(EventType, event_type_name, None)
        if et:
            await self._events.emit(Event(
                type=et,
                source="autonet",
                data=data or self.state.to_dict(),
            ))

    async def start(self) -> dict[str, Any]:
        """Start the autonet training service.

        Returns status dict for WS response.
        """
        if self.state.status == AutonetStatus.RUNNING:
            return {"status": "already_running"}

        self.state.status = AutonetStatus.STARTING
        log.info("Starting autonet service...")
        # Starting autonet IS joining the network: resolve the registry now
        # (never at boot).
        self.ensure_network_config()
        if self.state.jurisdiction_errors:
            self.state.status = AutonetStatus.ERROR
            self.state.last_error = "; ".join(self.state.jurisdiction_errors)
            return {"status": "error", "error": self.state.last_error}
        # The jurisdiction's charter defines the axes every world is built
        # on (docs/multi_jurisdiction.md decision 5): install it before the
        # nodes-side service exists.
        self._install_charter()
        self._loop = asyncio.get_running_loop()

        # Pre-flight cache refresh
        try:
            from ._cache import validate
            dao_registry = getattr(self.config, "dao_registry_address", "")
            if self.config.rpc_url and dao_registry:
                validate(
                    self.config.rpc_url, dao_registry,
                    __import__("atn").__version__,
                )
        except Exception:
            pass

        try:
            # Lazy import — nodes package is optional
            from nodes.common.config import load_config as load_autonet_config
            from nodes.service import AutonetService

            # Load autonet config (explicit path, AUTONET_CONFIG or ~/.autonet).
            # With a jurisdiction chosen, the atn side resolved every
            # address: the node's packaged (Shadownet) seed stays out.
            from .config import _selection_data_dir, jurisdiction_configured
            chosen = jurisdiction_configured(_selection_data_dir())
            config_path = self.config.config_path or None
            self._autonet_config = load_autonet_config(
                config_path, registry_fallback=not chosen)

            # Apply any ATN-level overrides
            if self.config.rpc_url:
                self._autonet_config.blockchain.rpc_url = self.config.rpc_url
            if self.config.chain_id:
                self._autonet_config.blockchain.chain_id = self.config.chain_id
            if self.config.private_key:
                self._autonet_config.blockchain.private_key = self.config.private_key
            elif self._agent_registry is not None:
                # No configured daemon key: the vault-held operator key
                # anchors, votes and claims (Decision 2026-10-27).
                try:
                    self._autonet_config.blockchain.private_key = (
                        self._agent_registry.operator_key())
                except Exception as e:
                    log.warning("operator key unavailable: %s", e)
            # Contract addresses: the atn side resolved them at join (packaged
            # registry, ATN_REGISTRY_URL or a launched jurisdiction's DAO
            # Registry), so its values win over the node's packaged seed,
            # empty ones too once a jurisdiction is chosen (a contract the
            # chosen DAO lacks must stay unset, not borrow Shadownet's).
            # rep_token must reach the close or every review weighs the same.
            bc = self._autonet_config.blockchain
            for atn_key, node_key in (("substrate_address", "substrate_address"),
                                      ("rep_token_address", "rep_token_address"),
                                      ("tool_bond_address", "tool_bond_address"),
                                      ("anchor_election_address",
                                       "anchor_election_address"),
                                      ("rep_claim_address", "rep_claim_address"),
                                      ("dao_registry_address", "dao_registry_address"),
                                      ("dao_address", "governor_address")):
                value = getattr(self.config, atn_key, "") or ""
                if value or chosen:
                    setattr(bc, node_key, value)
            if getattr(self.config, "deploy_block", 0) or chosen:
                bc.deploy_block = int(self.config.deploy_block or 0)
            # The jurisdiction's identity keys its gossip topic and its
            # parked substrate data (docs/multi_jurisdiction.md decision 2).
            self._autonet_config.jurisdiction_key = self.state.jurisdiction_key

            # Create the service. Note: AutonetService.start() blocks,
            # so we create here and start in an executor below. We wire
            # the epoch close subscriber BEFORE start so it's
            # registered when the library service gets constructed inside
            # start().
            self._service = AutonetService(self._autonet_config, data_dir=self._data_dir)
            # A charter change pinned to the previous anchor halts the
            # close and restarts the daemon onto the new axes.
            self._service.on_charter_change = self._on_charter_changed
            self._wire_library_epoch_subscriber()
            self._task = asyncio.create_task(self._run_service())

            self.state.status = AutonetStatus.RUNNING
            log.info("Autonet service started")
            # Start p2p agent advertisement alongside the training service
            self.start_p2p()
            # Wire P2P host to the service so training cycles can update gossip
            if self._p2p_host and self._service:
                self._service._p2p_host = self._p2p_host
            # Phase 10.2: wire substrate event-gossip across daemons.
            # Both the library service (constructed inside the executor-run
            # nodes-side start()) and the libp2p host (started in its
            # own thread, ready when its trio token is captured) come
            # up asynchronously, so spawn a small kicker that waits for
            # both before attaching.
            asyncio.create_task(self._wire_event_gossip_when_ready())
            asyncio.create_task(self._wire_chain_submission_when_ready())
            asyncio.create_task(self._wire_peer_attribution_when_ready())
            self._start_review_driver()
            await self._emit("AUTONET_STARTED")
            return {"status": "started"}

        except ImportError as e:
            self.state.status = AutonetStatus.ERROR
            self.state.last_error = f"Autonet nodes package not available: {e}"
            log.warning("Failed to start autonet: %s", e)
            return {"status": "error", "error": self.state.last_error}
        except Exception as e:
            self.state.status = AutonetStatus.ERROR
            self.state.last_error = str(e)
            log.exception("Failed to start autonet service")
            return {"status": "error", "error": str(e)}

    def _start_review_driver(self) -> None:
        if self._runtime is None or self._review_driver_task is not None:
            return
        try:
            from .review_driver import start_driver
            started = start_driver(self._runtime)
        except Exception:  # noqa: BLE001 (reviews wait, nothing else does)
            log.warning("review driver failed to start", exc_info=True)
            return
        if started is not None:
            self._review_driver, self._review_driver_task = started

    async def _stop_review_driver(self) -> None:
        task, self._review_driver_task = self._review_driver_task, None
        self._review_driver = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014
                pass

    def _build_agent_chain_resolver(self):
        """Return a zero-arg callable producing AgentChainIdentity list
        for every locally-registered agent with on-chain identity.

        Used by ChainSubmissionDriver to find per-agent signing keys.
        Skips agents that don't have a registered_on_chain identity —
        same gating rule as Phase 10.1's substrate-feed resolver.
        """
        registry = self._agent_registry
        if registry is None:
            return lambda: []

        try:
            from nodes.common.chain_submission_driver import AgentChainIdentity
        except Exception:
            return lambda: []

        def _resolve():
            out = []
            for defn, _status in registry.list_agents():
                if not defn.identity or not defn.identity.registered_on_chain:
                    continue
                addr = defn.identity.address or ""
                priv = registry.get_agent_key(defn.id) or ""
                if not addr or not priv:
                    continue
                out.append(AgentChainIdentity(address=addr, private_key=priv))
            return out

        return _resolve

    def _still_running(self, service: Any) -> bool:
        """True while ``service`` is the live service of a running bridge.
        A stop (or a restart with a new service) ends the wiring waiters."""
        return (self._service is service
                and self.state.status in (AutonetStatus.RUNNING,
                                          AutonetStatus.PAUSED))

    def _host_ready(self) -> bool:
        host = self._p2p_host
        ev = getattr(host, "_ready_event", None) if host is not None else None
        return ev is not None and ev.is_set()

    async def _wait_until_ready(
        self, ready: Callable[[], bool], what: str, service: Any, *,
        timeout: float = 30.0, poll: float = 0.5, max_poll: float = 10.0,
    ) -> bool:
        """Poll ``ready()`` until it holds or the service stops.

        The first ``timeout`` seconds poll every ``poll`` seconds. After
        that the wait logs one warning and keeps polling quietly, backing
        off to ``max_poll``. A slow host or library service delays wiring; it
        never cancels it. Returns False only when the service stopped."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        warned = False
        interval = poll
        while self._still_running(service):
            try:
                if ready():
                    return True
            except Exception:
                log.debug("%s readiness check failed", what, exc_info=True)
            if loop.time() >= deadline:
                if not warned:
                    log.warning("%s not ready after %.0fs; still waiting",
                                what, timeout)
                    warned = True
                interval = min(max_poll, interval * 2)
            await asyncio.sleep(interval)
        log.debug("%s wait ended: service stopped", what)
        return False

    async def _wire_peer_attribution_when_ready(self, timeout: float = 30.0) -> None:
        """Build the libp2p-PeerId ↔ on-chain-agent attribution table
        once the chain is reachable. No-op if substrate_address is
        unset (offline mode).

        The attribution table powers per-peer authorization: gossipped
        peers whose PeerId isn't on the chain are unregistered and can
        be ignored for consensus purposes.
        """
        rpb_cfg = self.config
        substrate_addr = (
            getattr(rpb_cfg, "substrate_address", "")
            or getattr(rpb_cfg, "rpb_contract_address", "")
        )
        if not substrate_addr or not rpb_cfg.rpc_url:
            log.debug("peer attribution: substrate_address/rpc_url unset, skipping")
            return

        # Wait for the libp2p host: the attribution table is stored on
        # it, which is where downstream lookups read it.
        if not await self._wait_until_ready(
                self._host_ready, "peer attribution (libp2p host)",
                self._service, timeout=timeout, poll=0.5):
            return

        def _blocking_build_attribution():
            """Synchronous web3 work — connect, build contract, refresh.
            Runs in an executor so the blocking RPC never stalls the event
            loop (which would freeze the WS server's accept path)."""
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(rpb_cfg.rpc_url))
            try:
                from nodes.common.contract_artifacts import load_abi
                abi = load_abi("Substrate")
            except FileNotFoundError:
                from atn.on_chain import SUBSTRATE_ABI as abi  # type: ignore

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(substrate_addr),
                abi=abi,
            )
            from nodes.common.peer_attribution import PeerAttribution
            attribution = PeerAttribution(contract)
            n = attribution.refresh()
            return attribution, n

        try:
            loop = asyncio.get_event_loop()
            attribution, n = await loop.run_in_executor(None, _blocking_build_attribution)
            log.info("Peer attribution wired: %d agents on chain", n)

            # Stash on the host so downstream consumers can use it.
            if self._p2p_host is not None:
                self._p2p_host._peer_attribution = attribution
        except Exception as e:
            log.warning("peer attribution attach failed: %s", e)

    async def _wire_chain_submission_when_ready(self, timeout: float = 30.0) -> None:
        """Attach chain submission once the federated-close driver is
        up. No-op if substrate_address is unset (offline mode)."""
        service = self._service
        if not service:
            return
        if not await self._wait_until_ready(
                lambda: getattr(service, "_federated_close_driver", None) is not None,
                "chain submission (federated-close driver)", service,
                timeout=timeout, poll=0.5):
            return
        # Only a daemon with a registered agent, or a registered operator
        # (the key that proposes and votes epoch elections for the
        # household), takes part in consensus and closes epochs; a
        # joined-only one follows the anchored state.
        registry = self._agent_registry
        operator_live = {"value": False, "checked": 0.0}

        def _operator_registered() -> bool:
            import time as _time
            now = _time.time()
            if operator_live["value"] or now - operator_live["checked"] < 60:
                return operator_live["value"]
            operator_live["checked"] = now
            try:
                from eth_account import Account
                from .on_chain import OnChainService
                svc = OnChainService(self.config)
                if not svc.available:
                    return False
                key_of = getattr(registry, "daemon_operator_key", None) \
                    or registry.operator_key
                addr = Account.from_key(key_of()).address
                w3 = svc._get_web3()
                operator_live["value"] = bool(
                    svc._get_contract(w3).functions.isLive(addr).call())
                if operator_live["value"] and hasattr(registry, "set_operator_registered"):
                    # External agents act as the operator: registered too.
                    registry.set_operator_registered(True)
            except Exception as e:
                log.debug("operator registration check failed: %s", e)
            return operator_live["value"]

        def _participating() -> bool:
            if registry is None:
                return False
            if any(d.identity and d.identity.registered_on_chain
                   for d, _ in registry.list_agents()):
                return True
            return _operator_registered()

        try:
            service._federated_close_driver.participating = _participating
        except AttributeError:          # a driver without the hook
            pass
        try:
            blob_resolver = self._build_libp2p_blob_resolver()
            ok = service.attach_chain_submission(
                agent_chain_resolver=self._build_agent_chain_resolver(),
                blob_resolver=blob_resolver,
                owner_wallet=getattr(self.config, "owner_wallet", "") or "",
            )
            if ok:
                log.info(
                    "Chain submission wired into AutonetService (blob_resolver=%s)",
                    "libp2p" if blob_resolver else "in-memory",
                )
        except Exception as e:
            log.warning("chain submission attach failed: %s", e)

    def _serve_disk_blobs(self, host) -> None:
        """Serve the daemon's on-disk blob stores (tool manifests and code,
        service specs) to peers, so what we published stays fetchable
        after a restart and while we are the only holder."""
        stores = []
        for name in ("tool_store", "service_store"):
            store = getattr(self._runtime, name, None) if self._runtime else None
            if store is None:
                continue
            try:
                stores.append(store._blob_store())
            except Exception as e:
                log.debug("%s blob store unavailable: %s", name, e)
        for blobs in stores:
            try:
                blobs.set_p2p_host(host)
            except Exception as e:
                log.debug("serving disk blobs failed: %s", e)

    def _share_search_model(self, host) -> None:
        """The search model's files are content-addressed blobs: serve the
        verified ones to peers (now, and again once the model is ready), and
        let a daemon still fetching them take them from peers when Hugging
        Face fails (atn/search_model.py). Bytes from a peer are checked
        against the pinned hashes like any download."""
        model = getattr(self._runtime, "search_model", None) if self._runtime else None
        if model is None or host is None:
            return
        try:
            model.serve_on(host)
            model.on_ready(lambda m, _h=host: m.serve_on(_h))
            from nodes.common.blob_resolver import LibP2PBlobResolver
            # A 90 MB file: give each peer minutes, not the 10 s default.
            resolver = LibP2PBlobResolver(host, fetch_timeout=600.0)

            def _get(digest: str, _r=resolver) -> bytes | None:
                # Not _r.get: that keeps every fetched blob in the host's
                # memory map, and these files are served from disk.
                peers = _r._known_peer_ids()
                return _r._fetch_from_peers(peers, digest) if peers else None

            async def _fetch(digest: str) -> bytes | None:
                return await asyncio.to_thread(_get, digest)
            model.peer_fetcher = _fetch
        except Exception as e:  # noqa: BLE001
            log.debug("sharing the search model failed: %s", e)

    def _build_libp2p_blob_resolver(self):
        """Build a LibP2PBlobResolver if the host is up; else None
        (driver falls back to InMemoryBlobResolver)."""
        host = self._p2p_host
        if host is None:
            return None
        try:
            from nodes.common.blob_resolver import LibP2PBlobResolver
        except Exception:
            return None
        try:
            return LibP2PBlobResolver(host)
        except Exception as e:
            log.debug("LibP2PBlobResolver build failed: %s", e)
            return None

    async def _wire_event_gossip_when_ready(self, timeout: float = 30.0) -> None:
        """Poll until the library service is up and the libp2p host is
        ready, then attach event gossip. Keeps waiting (with backoff,
        one warning) until wired or the service stops.

        The library service is constructed inside the nodes-side
        ``AutonetService.start()`` (executor-run); the libp2p host's
        trio token is captured inside its own thread. Both are
        asynchronous relative to this coroutine, so we poll briefly.
        """
        service = self._service
        if not service:
            return
        if not await self._wait_until_ready(
                lambda: (getattr(service, "_library", None) is not None
                         and self._host_ready()),
                "event gossip (library service and libp2p host)", service,
                timeout=timeout, poll=0.2):
            return
        host = self._p2p_host
        try:
            ok = service.attach_event_gossip(host)
            if ok:
                log.info("Event gossip wired into the library service")
        except Exception as e:
            log.warning("event gossip attach failed: %s", e)
        self._serve_disk_blobs(host)
        self._share_search_model(host)

        # Tool substrate: tool use is never gossiped (Decision 2026-11-07);
        # manifests still register.
        library = getattr(service, "_library", None)
        tool_store = getattr(self._runtime, "tool_store", None) if self._runtime else None
        if library is not None and tool_store is not None:
            def _manifest_sink(manifest: dict, author: str, _lib=library) -> None:
                _lib.submit_tool_manifest(manifest, agent_id=author)
            tool_store.manifest_sink = _manifest_sink

            # Network blob fetch (vetting + adoption): digest-verified
            # pull of foreign manifests/code over the libp2p blob
            # protocol. Absent host → local-only resolution.
            resolver = self._build_libp2p_blob_resolver()
            if resolver is not None:
                async def _fetch(digest: str, _r=resolver) -> bytes | None:
                    # resolver.get is a blocking cross-thread bridge into
                    # the trio host — keep it off the asyncio loop.
                    return await asyncio.to_thread(_r.get, digest)
                tool_store.blob_fetcher = _fetch
                service_store = getattr(self._runtime, "service_store", None)
                if service_store is not None:
                    service_store.blob_fetcher = _fetch

            # A new peer may hold code an approved adoption is waiting for
            # (#46): the retry loop watches the connected set.
            def _peers() -> list[str]:
                host = getattr(self, "_p2p_host", None)
                return host.get_connected_peers() if host is not None else []
            tool_store.peer_lister = _peers

            # Ratings for search and the app (Decision 2026-11-09): each
            # tool's plain means as of the latest close.
            def _rating(digest: str, _svc=service):
                drv = getattr(_svc, "_federated_close_driver", None)
                return drv.tool_rating(digest) if drv is not None else None
            tool_store.rating_provider = _rating
            # Sanctioned authors (docs/accountability.md): agents may not
            # propose adopting their tools.
            tool_store.sanction_check = lambda addr: self.sanctions.check(addr)

            # Tools registered while the network was down send their
            # registration now; ones an anchor already holds need none.
            drv = getattr(service, "_federated_close_driver", None)
            anchored = set(((drv.carry_over_maps() if drv is not None else {})
                            .get("tool_registrations") or {}))
            pushed = tool_store.push_all_manifests(
                skip=lambda digest: str(digest).lower() in anchored)
            log.info("ToolStore wired into the library service "
                     "(%d manifests backfilled)", pushed)

            # Network tool DISCOVERY: gossip ingest queues the manifest
            # digests of tools other households registered, but the blob
            # itself has to be pulled before `probe_tools` can rank it.
            # Do that here, off the gossip hot path: the ArtifactIndex is
            # derived daemon-local state (never gossiped, never anchored),
            # so a peer that is slow or offline only delays discovery — it
            # can never stall ingest or fork a close.
            self._manifest_pull_task = asyncio.create_task(
                self._pull_remote_manifests(library, tool_store))

            # Services, published and found the way tools are (Decision
            # 2026-11-10): specs gossiped whole on their own topic, kept
            # once the chain names them, searched locally.
            if self._service_catalog_task is None:
                self._service_catalog_task = asyncio.create_task(
                    self._run_service_catalog(str(library.rpb_address)))

    async def _run_service_catalog(self, jurisdiction_key: str) -> None:
        """Follow the jurisdiction's services: subscribe to their gossip,
        sync the catalog with the ServiceRegistry every ``SYNC_INTERVAL``,
        fetch the specs no announcement brought, and announce this
        daemon's own registered specs (at once on registration, then every
        ``ANNOUNCE_INTERVAL``)."""
        from . import service_catalog as sc
        from . import service_client

        runtime = self._runtime
        if runtime is None:
            return
        catalog = sc.catalog_for(runtime)
        service_client.set_catalog(catalog)
        topic = sc.topic_for(jurisdiction_key)
        host = self._p2p_host
        if host is not None:
            host.subscribe_topic_threadsafe(
                topic, lambda _t, data: catalog.accept_announcement(data))
        store = getattr(runtime, "service_store", None)

        def _announce(record) -> None:
            h = self._p2p_host
            if h is not None and record.service_id is not None:
                h.publish_topic_threadsafe(
                    topic, sc.announcement(record.spec, record.service_id))

        if store is not None:
            store.on_registered = _announce
        last_announce: float | None = None
        try:
            while True:
                try:
                    res = await asyncio.to_thread(catalog.sync_chain)
                    if res.get("error"):
                        log.debug("service catalog sync: %s", res["error"])
                    if store is not None:
                        kept = await sc.fetch_missing(catalog, store.spec_for)
                        if kept:
                            log.info("fetched %d service spec(s)", kept)
                    h = self._p2p_host
                    now = time.monotonic()
                    if h is not None and (last_announce is None
                                          or now - last_announce >= sc.ANNOUNCE_INTERVAL):
                        for msg in catalog.own_announcements():
                            h.publish_topic_threadsafe(topic, msg)
                        last_announce = now
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.debug("service catalog cycle failed", exc_info=True)
                await asyncio.sleep(sc.SYNC_INTERVAL)
        finally:
            service_client.clear_catalog(catalog)
            if store is not None and store.on_registered is _announce:
                store.on_registered = None

    async def _pull_remote_manifests(self, library, tool_store,
                                     interval: float = 10.0,
                                     batch: int = 64,
                                     concurrency: int = 8) -> None:
        """Drain queued remote manifest digests into the discovery index.

        Digest-verified by ``ToolStore.fetch_bytes`` (sha256 mismatch is
        dropped). Up to ``batch`` digests per cycle, ``concurrency`` at a
        time. A failed fetch stays queued and is retried with a doubling
        backoff (``LibraryService.take_pending_remote_manifests``); the
        anchored registrations re-queue any manifest gossip missed.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _one(digest: str) -> bool:
            async with sem:
                manifest = await tool_store.fetch_payload(digest)
            if not isinstance(manifest, dict):
                return False
            return bool(library.index_remote_manifest(manifest))

        while True:
            try:
                await asyncio.sleep(interval)
                digests = library.take_pending_remote_manifests(limit=batch)
                if not digests:
                    continue
                results = await asyncio.gather(
                    *(_one(d) for d in digests), return_exceptions=True)
                indexed = sum(1 for r in results if r is True)
                if indexed:
                    log.info("indexed %d remote tool manifest(s) for "
                             "network discovery", indexed)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug("remote manifest pull cycle failed", exc_info=True)

    def _wire_library_epoch_subscriber(self) -> None:
        """Forward the library service's epoch ticks to the EventBus.

        The subscriber runs in the autonet daemon's worker thread; we
        schedule the actual emit back on the main event loop so async
        EventBus consumers see it.
        """
        if not self._service or not self._events:
            return
        loop = asyncio.get_running_loop()

        def _on_close(record: dict) -> None:
            if not record.get("n_events"):
                return          # an empty local epoch: nothing to report
            # Slim payload: full record is durable on disk.
            payload = {
                "epoch_id": record.get("epoch_id"),
                "rpb_address": record.get("rpb_address"),
                "closed_at": record.get("closed_at"),
                "n_events": record.get("n_events", 0),
                "authoritative": record.get("authoritative", False),
                "scope": record.get("scope", "local"),
            }
            try:
                asyncio.run_coroutine_threadsafe(
                    self._emit("LIBRARY_EPOCH_CLOSED", payload),
                    loop,
                )
            except Exception:
                # If the loop is gone, swallow — we're shutting down.
                pass

        self._service.add_epoch_close_subscriber(_on_close)

        def _on_federated_close(fed) -> None:
            # The epoch's close fired here: commit this epoch's staked
            # review batches (atn/review_driver.py) before its anchor.
            driver = self._review_driver
            if driver is not None:
                try:
                    driver.on_local_close()
                except Exception:  # noqa: BLE001
                    log.debug("review driver close hook failed", exc_info=True)
            # FederatedCloseResult → the winner-reveal payload the
            # epochs UI animates (EPOCH_OBSERVABILITY_SPEC.md §4b).
            try:
                payload = {
                    "epoch_id": fed.epoch_id,
                    "n_batches": fed.n_batches,
                    "senders": [s.hex() for s in fed.senders],
                    "winner": fed.winner.hex() if fed.winner else "",
                    "is_us": bool(fed.is_winner),
                    "epoch_root": str(fed.close_result.get("epoch_root", "")),
                    "carry_cid": getattr(fed, "carry_cid", ""),
                    "total_mint": fed.close_result.get("total_mint", 0.0),
                    "agent_mint": fed.close_result.get("agent_mint", {}),
                }
            except Exception:
                log.exception("federated close payload build failed")
                return
            try:
                asyncio.run_coroutine_threadsafe(
                    self._emit("LIBRARY_FEDERATED_CLOSE", payload),
                    loop,
                )
            except Exception:
                pass

        self._service.add_federated_close_subscriber(_on_federated_close)

    async def _run_service(self) -> None:
        """Run the autonet service in a background thread."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._service.start)
        except Exception as e:
            self.state.status = AutonetStatus.ERROR
            self.state.last_error = str(e)
            self.state.errors += 1
            log.exception("Autonet service crashed")

    async def stop(self) -> dict[str, Any]:
        """Stop the autonet training service."""
        if self.state.status not in (AutonetStatus.RUNNING, AutonetStatus.PAUSED):
            return {"status": "not_running"}

        self.state.status = AutonetStatus.STOPPING
        log.info("Stopping autonet service...")

        try:
            if self._service:
                self._service.stop()
            if self._manifest_pull_task and not self._manifest_pull_task.done():
                self._manifest_pull_task.cancel()
                try:
                    await self._manifest_pull_task
                except asyncio.CancelledError:
                    pass
            self._manifest_pull_task = None
            if self._service_catalog_task and not self._service_catalog_task.done():
                self._service_catalog_task.cancel()
                try:
                    await self._service_catalog_task
                except asyncio.CancelledError:
                    pass
            self._service_catalog_task = None
            await self._stop_review_driver()
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

            self.stop_p2p()
            self.state.status = AutonetStatus.STOPPED
            log.info("Autonet service stopped")
            await self._emit("AUTONET_STOPPED")
            return {"status": "stopped"}
        except Exception as e:
            self.state.status = AutonetStatus.ERROR
            self.state.last_error = str(e)
            return {"status": "error", "error": str(e)}

    def get_status(self) -> dict[str, Any]:
        """Get current autonet status for WS response."""
        # Update resource metrics if service is running
        if self._service and self.state.status in (
            AutonetStatus.RUNNING, AutonetStatus.PAUSED
        ):
            try:
                svc_status = self._service.status()
                self.state.cycles_completed = svc_status.cycles_completed
                self.state.uptime_seconds = svc_status.uptime_seconds
                self.state.errors = svc_status.errors
                self.state.paused_reason = svc_status.paused_reason or ""
                # Sync state (service may have paused itself due to resources)
                if svc_status.state.value == "paused":
                    self.state.status = AutonetStatus.PAUSED
                elif svc_status.state.value == "running":
                    self.state.status = AutonetStatus.RUNNING
            except Exception:
                pass  # Service might be between cycles

        return self.state.to_dict()

    def connect_wallet(self, address: str) -> dict[str, Any]:
        """Register a wallet connection (from frontend wallet provider)."""
        self.state.wallet_connected = True
        self.state.wallet_address = address
        log.info("Wallet connected: %s", address[:10] + "...")
        return {
            "status": "connected",
            "address": address,
        }

    def disconnect_wallet(self) -> dict[str, Any]:
        """Disconnect the wallet."""
        self.state.wallet_connected = False
        self.state.wallet_address = ""
        log.info("Wallet disconnected")
        return {"status": "disconnected"}

    def set_chain(self, rpc_url: str, chain_id: int) -> dict[str, Any]:
        """Update the blockchain connection."""
        self.state.rpc_url = rpc_url
        self.state.chain_id = chain_id
        # If service is running, update its config too
        if self._autonet_config:
            self._autonet_config.blockchain.rpc_url = rpc_url
            self._autonet_config.blockchain.chain_id = chain_id
        log.info("Chain updated: %s (chain_id=%d)", rpc_url, chain_id)
        return {"status": "updated", "rpc_url": rpc_url, "chain_id": chain_id}
