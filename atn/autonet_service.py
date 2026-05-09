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
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

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
    # Resource usage (from ResourceMonitor)
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
    gas_symbol: str = "XTZ"
    gas_decimals: int = 18
    registry_address: str = ""
    token_address: str = ""
    economy_address: str = ""
    timelock_address: str = ""
    # Constitution CID loaded from registry
    constitution_cid: str = ""
    # Full constitution text (loaded from RPB.constitution() or local file)
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
            "registry_address": self.registry_address,
            "token_address": self.token_address,
            "economy_address": self.economy_address,
            "timelock_address": self.timelock_address,
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
        self._autonet_config = None  # Will hold nodes.common.config.AutonetConfig
        # Standards publication state (Story 3.2)
        self._published_standards_hash: str = ""
        self._published_tx_hash: str = ""
        self._user_contract_address: str = ""
        # Training data feed (Story 3.2) — ATN data dir for JSONL files
        self._data_dir = data_dir

        # Apply config values to state
        if config.wallet_address:
            self.state.wallet_connected = True
            self.state.wallet_address = config.wallet_address
        if config.rpc_url:
            self.state.rpc_url = config.rpc_url
        if config.chain_id:
            self.state.chain_id = config.chain_id

        # Always run the jurisdiction discovery hook so substrate_address
        # gets surfaced. The Governor-discovery body inside is a no-op
        # when dao_address/rpc_url are empty.
        self._discover_jurisdiction()

        # Subscribe to execution events for training data feed
        if self._events:
            self._events.subscribe(
                EventType.EXECUTION_COMPLETED,
                self._on_execution_completed,
            )

        # Runtime back-reference (set by Runtime.__init__)
        self._runtime = None

        # P2P agent advertisement
        self._p2p_host = None      # AutonetHost (lazy, trio-based)
        self._p2p_thread: threading.Thread | None = None
        self._p2p_stop = threading.Event()
        self._agent_registry = None  # Set by Runtime after init
        # Subscribe to agent lifecycle events for p2p advertisement refresh
        if self._events:
            for evt in (EventType.AGENT_REGISTERED, EventType.AGENT_UNREGISTERED,
                        EventType.AGENT_ACTIVATED, EventType.AGENT_DEACTIVATED):
                self._events.subscribe(evt, self._on_agent_changed)

        # Constitution CID loaded lazily from on-chain Registry
        self._constitution_loaded = False
        # Raw constitution text (loaded once, cached for prompt injection)
        self._constitution_text: str = ""

    def _discover_jurisdiction(self) -> None:
        """Discover all contract addresses from the DAO Governor at startup."""
        # Always surface substrate_address + gas token info from config —
        # they're seeded from registry.json and don't depend on the
        # on-chain Governor being reachable. (The block below may fail if
        # the Governor's Registry is unresponsive or the network is down;
        # these basics shouldn't depend on that.)
        self.state.substrate_address = getattr(
            self.config, "substrate_address", ""
        )
        self.state.gas_symbol = getattr(self.config, "gas_symbol", "XTZ") or "XTZ"
        self.state.gas_decimals = int(getattr(self.config, "gas_decimals", 18) or 18)
        try:
            from .on_chain import discover_jurisdiction
            discovered = discover_jurisdiction(
                self.config.rpc_url, self.config.dao_address,
            )
            # Populate config so OnChainService can use them
            self.config.rpb_contract_address = discovered.get("rpb", "")
            self.config.registry_address = discovered.get("registry", "")
            self.config.token_address = discovered.get("token", "")
            self.config.economy_address = discovered.get("economy", "")
            self.config.timelock_address = discovered.get("timelock", "")
            # Populate state for WS API
            self.state.dao_address = self.config.dao_address
            self.state.rpb_contract_address = self.config.rpb_contract_address
            # substrate_address is set from registry.json by the config
            # builder (not the on-chain Registry, since the Governor's
            # registry doesn't index substrate yet). Mirror it here.
            self.state.substrate_address = getattr(
                self.config, "substrate_address", ""
            )
            self.state.registry_address = self.config.registry_address
            self.state.token_address = self.config.token_address
            self.state.economy_address = self.config.economy_address
            self.state.timelock_address = self.config.timelock_address
            self.state.jurisdiction_name = discovered.get("jurisdiction_name", "")
            # Load constitution from registry if available
            cid = discovered.get("registry.rpb.prompt.current", "")
            if cid:
                self.state.constitution_cid = cid
                self._constitution_loaded = True
            log.info("Jurisdiction '%s' contracts discovered from %s",
                     self.state.jurisdiction_name, self.config.dao_address)
        except ImportError:
            log.info("Contract discovery skipped (install autonet-computer[network] for on-chain features)")
        except Exception as e:
            log.warning("Failed to discover jurisdiction from %s: %s",
                        self.config.dao_address, e)

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
            # Also include the connected wallet as a root agent if present
            if self.state.wallet_address and not any(
                a["address"].lower() == self.state.wallet_address.lower() for a in ads
            ):
                rpb_cfg = self.config.rpb if hasattr(self.config, 'rpb') else None
                ads.insert(0, {
                    "address": self.state.wallet_address,
                    "name": "root",
                    "description": "",
                    "agent_type": "orchestrator",
                    "model": getattr(rpb_cfg, "sponsor_model", "") if rpb_cfg else "",
                    "is_root": True,
                    "parent_address": "",
                    "registered_on_chain": False,
                    "is_sponsor": getattr(rpb_cfg, "sponsor_inference", False) if rpb_cfg else False,
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
        # Phase 12: a fresh install with no user-configured bootstrap
        # falls back to community-published bootstrap peers, so daemons
        # can join without manual multiaddr configuration.
        from nodes.common.peer_attribution import resolve_bootstrap_peers
        bootstrap = resolve_bootstrap_peers(cfg.p2p.bootstrap_peers if cfg else [])
        advertise_interval = cfg.p2p.capability_advertise_interval if cfg else 60

        node_id = f"atn-{self.state.wallet_address[:8]}" if self.state.wallet_address else "atn-daemon"
        cap = NodeCapability(peer_id="", node_id=node_id)

        host = AutonetHost(
            node_id=node_id,
            listen_port=listen_port,
            listen_host=listen_host,
            bootstrap_peers=bootstrap,
            capability=cap,
        )
        self._p2p_host = host
        self._p2p_stop.clear()

        # Wire sponsor-side inference handler (Path A)
        rpb_cfg = self.config.rpb if hasattr(self.config, 'rpb') else None
        if rpb_cfg and getattr(rpb_cfg, 'sponsor_inference', False):
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
            # No-chaining rule: reject requests that already came through RPB
            if request.get("via_rpb"):
                return {"error": "No-chaining: cannot re-route RPB inference through RPB"}

            model = request.get("model", rpb_cfg.sponsor_model or "")
            messages = request.get("messages", [])
            system = request.get("system", "")
            max_tokens = request.get("max_tokens", 4096)
            temperature = request.get("temperature", 0.0)
            tools_raw = request.get("tools", [])

            # Resolve provider — use configured sponsor provider or auto-resolve
            provider = service._resolve_sponsor_provider(rpb_cfg, model)
            if provider is None:
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

                # Serialize ProviderResponse back to dict for P2P transport
                result = {
                    "text": resp.text or "",
                    "model": resp.model or model,
                    "stop_reason": resp.stop_reason or "end_turn",
                    "usage": {
                        "input_tokens": resp.usage.input_tokens if resp.usage else 0,
                        "output_tokens": resp.usage.output_tokens if resp.usage else 0,
                    },
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
            from .providers.anthropic import AnthropicProvider
            from .credentials import CredentialStore
            creds = CredentialStore()
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

    def stop_p2p(self) -> None:
        """Stop the p2p background thread."""
        self._p2p_stop.set()
        if self._p2p_thread:
            self._p2p_thread.join(timeout=5)
            self._p2p_thread = None
        self._p2p_host = None

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

    async def load_constitution(self) -> str | None:
        """Load constitution text from on-chain RPB contract (once).

        Tries in order:
        1. RPB.constitution() — the actual text stored on-chain
        2. Local file fallback (constitution/v1_udhr.txt) for offline dev

        Called lazily on first state request or at startup.
        Returns the CID string or None.
        """
        if self._constitution_loaded:
            return self.state.constitution_cid or None
        self._constitution_loaded = True
        if not self.config.rpb_contract_address:
            # No chain config — fall back to local file
            self._load_constitution_text_local()
            self._push_constitution_to_training_feed()
            return None
        try:
            from .on_chain import OnChainService
            svc = OnChainService(self.config)
            # Try reading the full text directly from the RPB contract
            text = await svc.get_constitution_text()
            if text:
                self._constitution_text = text.strip()
                log.info("Constitution text loaded from RPB contract (%d chars)", len(self._constitution_text))
            else:
                log.debug("No constitution text on-chain, falling back to local file")
                self._load_constitution_text_local()
            # Also grab the CID from Registry if available
            if self.config.registry_address:
                cid = await svc.get_constitution_cid()
                if cid:
                    self.state.constitution_cid = cid
            self._push_constitution_to_training_feed()
            return self.state.constitution_cid or None
        except Exception as e:
            log.debug("Failed to load constitution from chain: %s", e)
            self._load_constitution_text_local()
            self._push_constitution_to_training_feed()
            return None

    def _push_constitution_to_training_feed(self) -> None:
        """Push loaded constitution text to the training feed and state."""
        # Always sync to state (for WS API)
        self.state.constitution_text = self._constitution_text
        if not self._constitution_text or not self._service:
            return
        feed = getattr(self._service, "_training_feed", None)
        if feed and hasattr(feed, "config"):
            feed.config.constitution_text = self._constitution_text
            log.debug("Constitution text pushed to training feed (%d chars)", len(self._constitution_text))

    def _load_constitution_text_local(self) -> None:
        """Load the raw constitution text from local file (offline fallback)."""
        try:
            from nodes.common.rpb_prompt import V1_PROMPT_FILE
            if V1_PROMPT_FILE.exists():
                self._constitution_text = V1_PROMPT_FILE.read_text(encoding="utf-8").strip()
                log.info("Constitution text loaded from local file (%d chars)", len(self._constitution_text))
            else:
                log.warning("No constitution text available at %s", V1_PROMPT_FILE)
        except Exception as e:
            log.debug("Failed to load constitution text: %s", e)

    @property
    def constitution_text(self) -> str:
        """The raw constitution text, empty if not loaded."""
        return self._constitution_text

    async def _on_execution_completed(self, event) -> None:
        """Handle EXECUTION_COMPLETED events from the EventBus.

        Story 3.2: When an agent execution completes, notify the training
        service so it knows new training data is available on disk.
        """
        data = event.data or {}
        agent_id = data.get("agent_id", "")
        execution_id = data.get("execution_id", "")
        status = data.get("status", "")

        if self._service:
            self._service.notify_execution(agent_id, execution_id, status)

    async def start(self) -> dict[str, Any]:
        """Start the autonet training service.

        Returns status dict for WS response.
        """
        if self.state.status == AutonetStatus.RUNNING:
            return {"status": "already_running"}

        self.state.status = AutonetStatus.STARTING
        log.info("Starting autonet service...")

        # Pre-flight cache refresh
        try:
            from ._cache import validate
            if self.config.rpc_url and self.config.registry_address:
                validate(
                    self.config.rpc_url, self.config.registry_address,
                    __import__("atn").__version__,
                )
        except Exception:
            pass

        try:
            # Lazy import — nodes package is optional
            from nodes.common.config import load_config as load_autonet_config
            from nodes.service import AutonetService

            # Load autonet config (discovers autonet.yaml)
            config_path = self.config.config_path or None
            self._autonet_config = load_autonet_config(config_path)

            # Apply any ATN-level overrides
            if self.config.rpc_url:
                self._autonet_config.blockchain.rpc_url = self.config.rpc_url
            if self.config.chain_id:
                self._autonet_config.blockchain.chain_id = self.config.chain_id
            if self.config.private_key:
                self._autonet_config.blockchain.private_key = self.config.private_key

            # Create the service. Note: AutonetService.start() blocks,
            # so we create here and start in an executor below. We wire
            # the WorldService close subscriber BEFORE start so it's
            # registered when the WorldService gets constructed inside
            # start().
            self._service = AutonetService(self._autonet_config, data_dir=self._data_dir)
            self._wire_world_epoch_subscriber()
            self._wire_substrate_identity_resolver()
            self._task = asyncio.create_task(self._run_service())

            self.state.status = AutonetStatus.RUNNING
            log.info("Autonet service started")
            # Start p2p agent advertisement alongside the training service
            self.start_p2p()
            # Wire P2P host to the service so training cycles can update gossip
            if self._p2p_host and self._service:
                self._service._p2p_host = self._p2p_host
            # Phase 6.4: wire the daemon's WorldService into the ATN
            # runtime's provider manager so the substrate provider can
            # probe it. Lazy-resolver: AutonetService creates its
            # WorldService inside start() (which runs in an executor),
            # so the resolver pulls the current value at lookup time
            # rather than capturing a possibly-None reference now.
            if self._runtime and hasattr(self._runtime, "providers") and self._service:
                svc_ref = self._service
                self._runtime.providers._world_service_resolver = (
                    lambda: getattr(svc_ref, "_world_service", None)
                )
            # Phase 10.2: wire substrate event-gossip across daemons.
            # Both the WorldService (constructed inside the executor-run
            # nodes-side start()) and the libp2p host (started in its
            # own thread, ready when its trio token is captured) come
            # up asynchronously, so spawn a small kicker that waits for
            # both before attaching.
            asyncio.create_task(self._wire_event_gossip_when_ready())
            asyncio.create_task(self._wire_chain_submission_when_ready())
            asyncio.create_task(self._wire_peer_attribution_when_ready())
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

        # Wait briefly for the libp2p host to come up. Attribution can
        # work without it (we can still read the chain) but storing
        # it on the host is the natural place for downstream lookups.
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            host_ready = (
                self._p2p_host is not None
                and getattr(self._p2p_host, "_ready_event", None) is not None
                and self._p2p_host._ready_event.is_set()
            )
            if host_ready:
                break
            await asyncio.sleep(0.5)

        try:
            from web3 import Web3
            from pathlib import Path
            import json as _json

            w3 = Web3(Web3.HTTPProvider(rpb_cfg.rpc_url))
            artifact_path = Path(
                "C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json"
            )
            try:
                with artifact_path.open("r", encoding="utf-8") as fh:
                    abi = _json.load(fh)["abi"]
            except FileNotFoundError:
                # Production install (no hardhat artifacts): fall back to the
                # inline ABI from atn.on_chain.
                from atn.on_chain import SUBSTRATE_ABI as abi  # type: ignore

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(substrate_addr),
                abi=abi,
            )

            from nodes.common.peer_attribution import PeerAttribution
            attribution = PeerAttribution(contract)
            n = attribution.refresh()
            log.info("Peer attribution wired: %d agents on chain", n)

            # Stash on the host so downstream consumers can use it.
            if self._p2p_host is not None:
                self._p2p_host._peer_attribution = attribution
        except Exception as e:
            log.warning("peer attribution attach failed: %s", e)

    async def _wire_chain_submission_when_ready(self, timeout: float = 30.0) -> None:
        """Attach chain submission once the federated-close driver is
        up. No-op if substrate_address is unset (offline mode)."""
        deadline = asyncio.get_event_loop().time() + timeout
        service = self._service
        if not service:
            return
        while asyncio.get_event_loop().time() < deadline:
            ready = getattr(service, "_federated_close_driver", None) is not None
            if ready:
                break
            await asyncio.sleep(0.5)
        else:
            log.debug("chain submission wait timed out (federated-close driver not ready)")
            return
        try:
            blob_resolver = self._build_libp2p_blob_resolver()
            ok = service.attach_chain_submission(
                agent_chain_resolver=self._build_agent_chain_resolver(),
                blob_resolver=blob_resolver,
            )
            if ok:
                log.info(
                    "Chain submission wired into AutonetService (blob_resolver=%s)",
                    "libp2p" if blob_resolver else "in-memory",
                )
        except Exception as e:
            log.warning("chain submission attach failed: %s", e)

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
        """Poll until the WorldService is up and the libp2p host is
        ready, then attach event gossip. No-ops on timeout.

        WorldService is constructed inside the nodes-side
        ``AutonetService.start()`` (executor-run); the libp2p host's
        trio token is captured inside its own thread. Both are
        asynchronous relative to this coroutine, so we poll briefly.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        host = self._p2p_host
        service = self._service
        if not service:
            return
        while asyncio.get_event_loop().time() < deadline:
            ws_ready = getattr(service, "_world_service", None) is not None
            host_ready = (
                host is not None
                and getattr(host, "_ready_event", None) is not None
                and host._ready_event.is_set()
            )
            if ws_ready and host_ready:
                break
            await asyncio.sleep(0.2)
        else:
            log.debug("event gossip wait timed out (ws or host not ready)")
            return
        try:
            ok = service.attach_event_gossip(host)
            if ok:
                log.info("Event gossip wired into WorldService")
        except Exception as e:
            log.warning("event gossip attach failed: %s", e)

    def _wire_substrate_identity_resolver(self) -> None:
        """Pass an identity resolver to the substrate feed so each
        work unit is attributed to the actual agent's 0x address
        rather than collapsed under a single "daemon" author.

        Skips agents without on-chain identity — they have no
        network impact (no 0x address means no mint attribution).
        """
        if not self._service or not self._agent_registry:
            return

        try:
            from nodes.common.world_substrate_feed import ResolvedAgentIdentity
        except ImportError:
            return

        registry = self._agent_registry

        def _resolve(local_agent_id: str):
            defn = registry.get_agent(local_agent_id)
            if defn is None or defn.identity is None:
                return None
            return ResolvedAgentIdentity(
                address=defn.identity.address or "",
                registered_on_chain=bool(defn.identity.registered_on_chain),
            )

        self._service.set_substrate_identity_resolver(_resolve)

    def _wire_world_epoch_subscriber(self) -> None:
        """Forward WorldService epoch closes to the EventBus.

        The subscriber runs in the autonet daemon's worker thread (under
        the WorldService lock); we schedule the actual emit back on the
        main event loop so async EventBus consumers see it.
        """
        if not self._service or not self._events:
            return
        loop = asyncio.get_running_loop()

        def _on_close(record: dict) -> None:
            # Slim payload: full record is durable on disk.
            payload = {
                "epoch_id": record.get("epoch_id"),
                "rpb_address": record.get("rpb_address"),
                "closed_at": record.get("closed_at"),
                "n_events": record.get("n_events", 0),
                "total_mint": record.get("total_mint", 0.0),
                "total_novelty": record.get("total_novelty", 0.0),
                "agent_mint": record.get("agent_mint", {}),
                "authoritative": record.get("authoritative", False),
                "scope": record.get("scope", "local"),
            }
            try:
                asyncio.run_coroutine_threadsafe(
                    self._emit("WORLD_EPOCH_CLOSED", payload),
                    loop,
                )
            except Exception:
                # If the loop is gone, swallow — we're shutting down.
                pass

        self._service.add_epoch_close_subscriber(_on_close)

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
                # Pull resource snapshot if available
                if self._service._resource_monitor:
                    snap = self._service._resource_monitor.last_snapshot
                    if snap:
                        self.state.cpu_percent = snap.cpu_percent
                        self.state.memory_mb = snap.memory_mb
                        self.state.gpu_memory_mb = snap.gpu_memory_mb
                        try:
                            import torch
                            self.state.gpu_available = torch.cuda.is_available()
                        except ImportError:
                            self.state.gpu_available = False
                # Pull training feed metrics
                self._sync_training_metrics()
            except Exception:
                pass  # Service might be between cycles

        return self.state.to_dict()

    def _sync_training_metrics(self) -> None:
        """Pull real-time training metrics from the training feed into state."""
        feed = getattr(self._service, "_training_feed", None)
        if not feed:
            return
        self.state.pending_events = feed._pending_events
        self.state.training_segments = getattr(feed, "_last_segment_count", 0)
        self.state.last_cycle_time = feed._last_cycle_time

        # Per-cycle metrics from the last training run
        metrics = getattr(feed, "_last_metrics", None)
        if metrics:
            loss = metrics.get("loss", 0.0)
            self.state.training_loss = loss
            self.state.training_batches = metrics.get("num_batches", 0)
            self.state.training_cosine_sim = metrics.get("cosine_similarity", 0.0)
            # Append to loss history if this is a new cycle
            cycle_count = feed._cycles_completed
            if cycle_count > len(self.state.loss_history):
                self.state.loss_history.append(round(loss, 6))
                if len(self.state.loss_history) > self.state._MAX_LOSS_HISTORY:
                    self.state.loss_history = self.state.loss_history[-self.state._MAX_LOSS_HISTORY:]

        # Behavioral profile hash
        profile = getattr(feed, "_profile", None)
        if profile:
            profile_hash = getattr(profile, "hash", None)
            if callable(profile_hash):
                try:
                    self.state.profile_hash = profile_hash()
                except Exception:
                    pass
            elif isinstance(profile_hash, str):
                self.state.profile_hash = profile_hash

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

    # ------------------------------------------------------------------
    # Standards publication (Story 3.2)
    # ------------------------------------------------------------------

    def get_standards(self, user_profile=None) -> dict[str, Any]:
        """Get current standards and on-chain publication status.

        Standards are derived from the user profile (populated during
        onboarding or conversation).  The hash can optionally be
        published on-chain via `publish_standards()`.

        Args:
            user_profile: UserProfile instance (from runtime.user_profile)
        """
        standards: list[dict[str, Any]] = []
        if user_profile:
            profile = user_profile.get_profile()
            standards = profile.standards or []

        # Compute the hash that would go on-chain
        standards_hash = self._compute_standards_hash(standards)

        return {
            "standards": standards,
            "standards_hash": standards_hash,
            "on_chain": {
                "published": self.state.wallet_connected and bool(self._published_standards_hash),
                "tx_hash": self._published_tx_hash,
                "published_hash": self._published_standards_hash,
                "matches_current": self._published_standards_hash == standards_hash,
                "user_contract": self._user_contract_address,
            },
        }

    async def publish_standards(self, user_profile=None, private_key: str = "") -> dict[str, Any]:
        """Publish standards hash on-chain by creating a user contract.

        Requires:
          - Wallet connected (address set)
          - RPC URL configured
          - Private key (passed from client-side wallet, NOT stored)
          - Standards exist in user profile

        Args:
            user_profile: UserProfile instance
            private_key: Hex private key for signing the transaction
        """
        if not self.state.wallet_connected:
            return {"status": "error", "error": "Wallet not connected"}
        if not self.state.rpc_url:
            return {"status": "error", "error": "No RPC URL configured"}
        if not private_key:
            return {"status": "error", "error": "Private key required for signing"}

        # Get standards
        standards: list[dict[str, Any]] = []
        if user_profile:
            profile = user_profile.get_profile()
            standards = profile.standards or []

        if not standards:
            return {"status": "error", "error": "No standards defined. Complete onboarding first."}

        standards_hash = self._compute_standards_hash(standards)
        standards_hash_bytes = bytes.fromhex(standards_hash)

        try:
            from web3 import Web3
            from eth_account import Account

            w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
            if not w3.is_connected():
                return {"status": "error", "error": f"Cannot connect to {self.state.rpc_url}"}

            account = Account.from_key(private_key)
            if account.address.lower() != self.state.wallet_address.lower():
                return {"status": "error", "error": "Private key does not match connected wallet"}

            # Load the Autonet contract ABI
            autonet_address = self._get_autonet_contract_address()
            if not autonet_address:
                return {"status": "error", "error": "Autonet contract address not configured"}

            abi = self._load_contract_abi("Autonet")
            if not abi:
                return {"status": "error", "error": "Autonet contract ABI not found"}

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(autonet_address),
                abi=abi,
            )

            # Build and send createUserContract transaction
            nonce = w3.eth.get_transaction_count(account.address)
            tx = contract.functions.createUserContract(
                standards_hash_bytes
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 2_000_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": self.state.chain_id or w3.eth.chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                # Parse UserContractCreated event to get user contract address
                user_contract_addr = ""
                try:
                    logs = contract.events.UserContractCreated().process_receipt(receipt)
                    if logs:
                        user_contract_addr = logs[0]["args"]["userContract"]
                except Exception:
                    pass

                self._published_standards_hash = standards_hash
                self._published_tx_hash = tx_hash.hex()
                self._user_contract_address = user_contract_addr

                log.info("Standards published on-chain: tx=%s, user_contract=%s",
                         self._published_tx_hash, user_contract_addr)

                await self._emit("AUTONET_STATUS", {
                    "action": "standards_published",
                    "tx_hash": self._published_tx_hash,
                    "standards_hash": standards_hash,
                    "user_contract": user_contract_addr,
                })

                return {
                    "status": "published",
                    "tx_hash": self._published_tx_hash,
                    "standards_hash": standards_hash,
                    "user_contract": user_contract_addr,
                }
            else:
                return {"status": "error", "error": "Transaction reverted"}

        except ImportError:
            return {"status": "error", "error": "web3 package not installed. Install with: pip install web3"}
        except Exception as e:
            log.exception("Failed to publish standards")
            return {"status": "error", "error": str(e)}

    @staticmethod
    def _compute_standards_hash(standards: list[dict[str, Any]]) -> str:
        """Compute a deterministic bytes32 hash of the standards list.

        The hash is a keccak256 of the JSON-serialized standards,
        matching the Solidity bytes32 format.
        """
        if not standards:
            return "0" * 64
        # Canonical JSON: sorted keys, no whitespace
        canonical = json.dumps(standards, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _get_autonet_contract_address(self) -> str | None:
        """Get the deployed Autonet contract address from config."""
        return self._get_contract_address("autonet")

    def _load_deployment_addresses(self) -> dict[str, str]:
        """Load contract addresses from deployment-addresses.json."""
        from pathlib import Path
        candidates = [
            Path("deployment-addresses.json"),
            Path(__file__).parent.parent / "deployment-addresses.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    with open(path) as f:
                        return json.load(f)
                except Exception:
                    continue
        return {}

    def _load_contract_abi(self, contract_name: str) -> list | None:
        """Load a contract ABI from the artifacts directory."""
        from pathlib import Path

        base = Path(__file__).parent.parent / "artifacts" / "contracts"
        base2 = Path("artifacts") / "contracts"

        # Search all subdirectories under artifacts/contracts/
        subdirs = ["economic", "governance", "core", "tokens", "bridge",
                    "rollup", "utils", "interfaces", "mocks"]
        candidates = []
        for sd in subdirs:
            candidates.append(base / sd / f"{contract_name}.sol" / f"{contract_name}.json")
            candidates.append(base2 / sd / f"{contract_name}.sol" / f"{contract_name}.json")
        # Also check flat locations
        candidates.append(base.parent / f"{contract_name}.json")
        candidates.append(Path("artifacts") / f"{contract_name}.json")

        for path in candidates:
            if path.exists():
                try:
                    with open(path) as f:
                        artifact = json.load(f)
                    return artifact.get("abi", [])
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # Earnings dashboard (Story 3.5)
    # ------------------------------------------------------------------

    async def get_earnings(self) -> dict[str, Any]:
        """Get the user's ATN earnings summary.

        Returns local training metrics and, if wallet+contract are
        available, on-chain epoch reward data.
        """
        # Local training stats (always available)
        local = {
            "cycles_completed": self.state.cycles_completed,
            "uptime_seconds": self.state.uptime_seconds,
            "status": self.state.status.value,
        }

        # On-chain earnings (requires wallet + contract)
        on_chain: dict[str, Any] = {
            "available": False,
            "balance": "0",
            "total_burned": "0",
            "current_epoch": 0,
            "claimable_rewards": [],
        }

        if (self.state.wallet_connected
                and self.state.rpc_url
                and self._get_autonet_contract_address()):
            try:
                on_chain = await self._fetch_on_chain_earnings()
            except Exception as e:
                log.debug("Failed to fetch on-chain earnings: %s", e)

        return {
            "local": local,
            "on_chain": on_chain,
        }

    async def _fetch_on_chain_earnings(self) -> dict[str, Any]:
        """Fetch earnings data from the Autonet contract."""
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
        if not w3.is_connected():
            return {"available": False, "error": "RPC not reachable"}

        autonet_address = self._get_autonet_contract_address()
        abi = self._load_contract_abi("Autonet")
        if not autonet_address or not abi:
            return {"available": False, "error": "Contract not configured"}

        contract = w3.eth.contract(
            address=Web3.to_checksum_address(autonet_address),
            abi=abi,
        )

        wallet = Web3.to_checksum_address(self.state.wallet_address)

        # Fetch key data points
        balance = contract.functions.balanceOf(wallet).call()
        current_epoch = contract.functions.currentEpoch().call()
        total_burned = contract.functions.totalInferenceBurned().call()
        user_burned = contract.functions.userInferenceBurned(wallet).call()

        # Check user contract
        user_contract = contract.functions.getUserContract(wallet).call()
        has_user_contract = user_contract != "0x" + "0" * 40

        # Scan recent epochs for claimable rewards
        claimable: list[dict[str, Any]] = []
        scan_start = max(1, current_epoch - 10)  # Last 10 epochs
        for epoch_id in range(scan_start, current_epoch + 1):
            try:
                epoch_stats = contract.functions.getEpochStats(epoch_id).call()
                total_usage, budget, finalized, _ = epoch_stats
                if not finalized or budget == 0:
                    continue

                # Check all services for this user's attested usage
                service_ids = contract.functions.getAllServiceIds().call()
                for sid in service_ids:
                    attester_usage = contract.functions.getAttesterUsage(
                        epoch_id, sid, wallet
                    ).call()
                    if attester_usage == 0:
                        continue

                    # Check if already claimed
                    already_claimed = contract.functions.hasClaimedReward(
                        sid, epoch_id, wallet
                    ).call()
                    if already_claimed:
                        continue

                    # Estimate reward
                    epoch_reward = contract.functions.getServiceEpochReward(
                        sid, epoch_id
                    ).call()
                    reward_total, claimed_amount, _ = epoch_reward
                    if reward_total == 0:
                        continue

                    service_epoch_usage = contract.functions.epochUsage(
                        epoch_id, sid
                    ).call() if hasattr(contract.functions, 'epochUsage') else total_usage

                    estimated = (attester_usage * reward_total) // service_epoch_usage if service_epoch_usage > 0 else 0

                    claimable.append({
                        "epoch": epoch_id,
                        "service_id": sid.hex() if isinstance(sid, bytes) else str(sid),
                        "attested_units": attester_usage,
                        "estimated_reward_wei": str(estimated),
                        "estimated_reward": str(estimated / 10**18) if estimated > 0 else "0",
                    })
            except Exception:
                continue  # Skip epochs with errors

        return {
            "available": True,
            "balance": str(balance),
            "balance_formatted": f"{balance / 10**18:.4f}",
            "current_epoch": current_epoch,
            "total_burned": str(total_burned),
            "user_burned": str(user_burned),
            "has_user_contract": has_user_contract,
            "user_contract": user_contract if has_user_contract else "",
            "claimable_rewards": claimable,
        }

    async def claim_reward(self, epoch_id: int, service_id: str,
                           private_key: str) -> dict[str, Any]:
        """Claim participant reward for an epoch."""
        if not self.state.wallet_connected:
            return {"status": "error", "error": "Wallet not connected"}
        if not private_key:
            return {"status": "error", "error": "Private key required"}

        try:
            from web3 import Web3
            from eth_account import Account

            w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
            account = Account.from_key(private_key)
            autonet_address = self._get_autonet_contract_address()
            abi = self._load_contract_abi("Autonet")

            if not autonet_address or not abi:
                return {"status": "error", "error": "Contract not configured"}

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(autonet_address),
                abi=abi,
            )

            sid_bytes = bytes.fromhex(service_id) if isinstance(service_id, str) else service_id

            nonce = w3.eth.get_transaction_count(account.address)
            tx = contract.functions.claimParticipantReward(
                sid_bytes, epoch_id
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 500_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": self.state.chain_id or w3.eth.chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                log.info("Reward claimed: epoch=%d, tx=%s", epoch_id, tx_hash.hex())
                return {
                    "status": "claimed",
                    "tx_hash": tx_hash.hex(),
                    "epoch": epoch_id,
                }
            else:
                return {"status": "error", "error": "Transaction reverted"}

        except Exception as e:
            log.exception("Failed to claim reward")
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # Proposals & governance (Stories 3.6 / 3.7)
    # ------------------------------------------------------------------

    async def get_proposals(self) -> dict[str, Any]:
        """List evolution proposals from the EvolutionProposal contract."""
        if not self.state.wallet_connected or not self.state.rpc_url:
            return {"available": False, "proposals": []}

        try:
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
            if not w3.is_connected():
                return {"available": False, "proposals": []}

            ep_address = self._get_contract_address("evolution_proposal")
            abi = self._load_contract_abi("EvolutionProposal")
            if not ep_address or not abi:
                return {"available": False, "proposals": []}

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(ep_address), abi=abi,
            )

            count = contract.functions.getProposalCount().call()
            proposals: list[dict[str, Any]] = []

            for i in range(min(count, 50)):  # Cap at 50 most recent
                pid = count - i  # Newest first
                try:
                    result = contract.functions.getProposal(pid).call()
                    proposer, content_cid, stake, status, created_at, trial_budget, eval_count, eval_score, dao_id = result
                    status_names = ["proposed", "evaluating", "trial", "adopted", "rejected"]
                    proposals.append({
                        "id": pid,
                        "proposer": proposer,
                        "content_cid": content_cid,
                        "stake": str(stake),
                        "status": status_names[status] if status < len(status_names) else "unknown",
                        "created_at": created_at,
                        "trial_budget": str(trial_budget),
                        "evaluation_count": eval_count,
                        "evaluation_score": eval_score,
                        "dao_proposal_id": dao_id,
                    })
                except Exception:
                    continue

            return {"available": True, "proposals": proposals}

        except ImportError:
            return {"available": False, "error": "web3 not installed"}
        except Exception as e:
            log.debug("Failed to fetch proposals: %s", e)
            return {"available": False, "proposals": []}

    async def get_governance(self) -> dict[str, Any]:
        """Get governance overview from AutonetDAO."""
        if not self.state.wallet_connected or not self.state.rpc_url:
            return {"available": False, "proposals": []}

        try:
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(self.state.rpc_url))
            if not w3.is_connected():
                return {"available": False, "proposals": []}

            dao_address = self._get_contract_address("dao")
            abi = self._load_contract_abi("AutonetDAO")
            if not dao_address or not abi:
                return {"available": False, "proposals": []}

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(dao_address), abi=abi,
            )

            next_id = contract.functions.nextProposalId().call()
            proposals: list[dict[str, Any]] = []

            for pid in range(max(1, next_id - 20), next_id):
                try:
                    p = contract.functions.proposals(pid).call()
                    state = contract.functions.getProposalState(pid).call()
                    state_names = ["pending", "active", "canceled", "defeated",
                                   "succeeded", "queued", "expired", "executed"]
                    proposals.append({
                        "id": p[0],
                        "proposer": p[1],
                        "description": p[2],
                        "start_block": p[3],
                        "end_block": p[4],
                        "for_votes": str(p[5]),
                        "against_votes": str(p[6]),
                        "executed": p[7],
                        "canceled": p[8],
                        "state": state_names[state] if state < len(state_names) else "unknown",
                    })
                except Exception:
                    continue

            return {"available": True, "proposals": proposals}

        except ImportError:
            return {"available": False, "error": "web3 not installed"}
        except Exception as e:
            log.debug("Failed to fetch governance: %s", e)
            return {"available": False, "proposals": []}

    def _get_contract_address(self, name: str) -> str | None:
        """Get a named contract address from config or deployment-addresses.json.

        Name mapping:
          autonet → ATNToken (main Autonet entry-point uses ATN Token)
          dao → AutonetDAO
          evolution_proposal → (not yet deployed separately)
          staking → ParticipantStaking
          Any key from deployment-addresses.json
        """
        # Mapping from logical names to deployment-addresses.json keys
        name_map = {
            "autonet": "Autonet",
            "dao": "AutonetDAO",
            "rpb": "RPB",
            "rpb_factory": "RPBFactory",
            "staking": "ParticipantStaking",
            "shard_registry": "ModelShardRegistry",
            "task": "TaskContract",
            "results": "ResultsRewards",
            "bridge": "AnchorBridge",
            "disputes": "DisputeManager",
            "forced_errors": "ForcedErrorRegistry",
            "inference_factory": "InferenceProviderFactory",
        }

        # 1. Check autonet YAML config (blockchain.contracts section)
        if self._autonet_config and hasattr(self._autonet_config, "blockchain"):
            bc = self._autonet_config.blockchain
            # Check direct attribute
            attr_name = f"{name}_contract"
            addr = getattr(bc, attr_name, None)
            if addr:
                return addr
            # Check contracts dict if it exists
            contracts = getattr(bc, "contracts", None)
            if contracts:
                json_key = name_map.get(name, name)
                if json_key in contracts:
                    return contracts[json_key]

        # 2. Check deployment-addresses.json
        addresses = self._load_deployment_addresses()
        if addresses:
            json_key = name_map.get(name, name)
            if json_key in addresses:
                return addresses[json_key]

        # 3. Check ATN config
        contract_addr = getattr(self.config, f"{name}_contract", None)
        return contract_addr

    # ------------------------------------------------------------------
    # Alignment dashboard (Story 3.8)
    # ------------------------------------------------------------------

    def get_alignment(self, user_profile=None) -> dict[str, Any]:
        """Get alignment status between user standards and network behavior.

        Shows how well the node's training activity reflects the user's
        stated standards and values.  Without live training data, returns
        the structural alignment (standards exist, training active, etc.).
        """
        standards: list[dict[str, Any]] = []
        if user_profile:
            profile = user_profile.get_profile()
            standards = profile.standards or []

        # Alignment indicators
        has_standards = len(standards) > 0
        is_training = self.state.status.value == "running"
        has_wallet = self.state.wallet_connected
        has_user_contract = bool(self._user_contract_address)
        standards_published = bool(self._published_standards_hash)

        # Compute an alignment score (0-100)
        # Each factor contributes to overall alignment readiness
        score = 0
        factors: list[dict[str, Any]] = []

        if has_standards:
            score += 25
            factors.append({"name": "Standards defined", "met": True,
                           "description": f"{len(standards)} standards in your profile"})
        else:
            factors.append({"name": "Standards defined", "met": False,
                           "description": "Complete onboarding to define your standards"})

        if standards_published:
            score += 25
            factors.append({"name": "Standards published", "met": True,
                           "description": "Your standards hash is recorded on-chain"})
        elif has_wallet:
            factors.append({"name": "Standards published", "met": False,
                           "description": "Publish your standards to establish on-chain identity"})
        else:
            factors.append({"name": "Standards published", "met": False,
                           "description": "Connect wallet first, then publish standards"})

        if is_training:
            score += 25
            factors.append({"name": "Training active", "met": True,
                           "description": f"{self.state.cycles_completed} cycles completed"})
        else:
            factors.append({"name": "Training active", "met": False,
                           "description": "Start training to contribute to the network"})

        if has_user_contract:
            score += 25
            factors.append({"name": "Node registered", "met": True,
                           "description": "Your node has an on-chain identity"})
        elif has_wallet:
            factors.append({"name": "Node registered", "met": False,
                           "description": "Publish standards to register your node"})
        else:
            factors.append({"name": "Node registered", "met": False,
                           "description": "Connect wallet to register on-chain"})

        return {
            "score": score,
            "factors": factors,
            "standards_count": len(standards),
            "training_active": is_training,
            "wallet_connected": has_wallet,
            "node_registered": has_user_contract,
        }

    # ------------------------------------------------------------------
    # Data capture & privacy (Story 3.4)
    # ------------------------------------------------------------------

    def get_capture_config(self) -> dict[str, Any]:
        """Get the current capture and privacy configuration."""
        cfg = self._autonet_config or self._load_autonet_config_readonly()
        return {
            "capture": {
                "enabled_sources": list(cfg.capture.enabled_sources),
                "fps_cap": cfg.capture.fps_cap,
                "resolution": list(cfg.capture.resolution),
                "screen_monitor": cfg.capture.screen_monitor,
                "browser_scrub_pii": cfg.capture.browser_scrub_pii,
                "browser_exclude_patterns": list(cfg.capture.browser_exclude_patterns),
            },
            "privacy": {
                "exclude_apps": list(cfg.privacy.exclude_apps),
                "blur_regions": list(cfg.privacy.blur_regions),
                "scrub_pii": cfg.privacy.scrub_pii,
            },
        }

    def set_capture_config(self, capture: dict[str, Any] | None = None,
                           privacy: dict[str, Any] | None = None) -> dict[str, Any]:
        """Update capture and/or privacy configuration."""
        cfg = self._autonet_config or self._load_autonet_config_readonly()

        if capture:
            if "enabled_sources" in capture:
                cfg.capture.enabled_sources = list(capture["enabled_sources"])
            if "fps_cap" in capture:
                cfg.capture.fps_cap = int(capture["fps_cap"])
            if "resolution" in capture:
                cfg.capture.resolution = list(capture["resolution"])
            if "screen_monitor" in capture:
                cfg.capture.screen_monitor = int(capture["screen_monitor"])
            if "browser_scrub_pii" in capture:
                cfg.capture.browser_scrub_pii = bool(capture["browser_scrub_pii"])
            if "browser_exclude_patterns" in capture:
                cfg.capture.browser_exclude_patterns = list(capture["browser_exclude_patterns"])

        if privacy:
            if "exclude_apps" in privacy:
                cfg.privacy.exclude_apps = list(privacy["exclude_apps"])
            if "blur_regions" in privacy:
                cfg.privacy.blur_regions = list(privacy["blur_regions"])
            if "scrub_pii" in privacy:
                cfg.privacy.scrub_pii = bool(privacy["scrub_pii"])

        self._autonet_config = cfg
        self._save_autonet_config(cfg)
        log.info("Capture/privacy config updated")
        return self.get_capture_config()

    def enumerate_sources(self) -> dict[str, Any]:
        """Enumerate available capture sources (screens, browsers).

        Returns a list of available sources the user can select from.
        """
        sources: list[dict[str, Any]] = []

        # Enumerate screens via mss
        try:
            import mss
            with mss.mss() as sct:
                for i, mon in enumerate(sct.monitors):
                    if i == 0:
                        continue  # monitor 0 is the "all monitors" virtual screen
                    sources.append({
                        "type": "screen",
                        "id": f"screen_{i}",
                        "name": f"Screen {i}",
                        "monitor_index": i,
                        "width": mon["width"],
                        "height": mon["height"],
                        "left": mon["left"],
                        "top": mon["top"],
                    })
        except ImportError:
            log.debug("mss not available — screen enumeration disabled")
        except Exception as e:
            log.debug("Screen enumeration failed: %s", e)

        # Browser source (always available if the relay is configured)
        sources.append({
            "type": "browser",
            "id": "browser",
            "name": "Browser Activity",
            "description": "Captures browsing activity via CDP relay",
        })

        return {"sources": sources}

    def _load_autonet_config_readonly(self):
        """Load autonet config without storing it (for read-only access)."""
        try:
            from nodes.common.config import load_config as load_autonet_config
            config_path = self.config.config_path or None
            return load_autonet_config(config_path)
        except ImportError:
            return None

    def _save_autonet_config(self, cfg) -> None:
        """Save autonet config back to YAML file."""
        try:
            import yaml
            from pathlib import Path

            # Find the config file path
            config_path = self.config.config_path
            if not config_path:
                import os
                config_path = os.environ.get("AUTONET_CONFIG")
            if not config_path:
                candidates = [
                    Path("autonet.yaml"),
                    Path.home() / ".autonet" / "config.yaml",
                ]
                for c in candidates:
                    if c.exists():
                        config_path = str(c)
                        break

            if not config_path:
                log.warning("No config file found to save to")
                return

            # Read existing YAML, update capture/privacy sections, write back
            path = Path(config_path)
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

            raw["capture"] = {
                "enabled_sources": cfg.capture.enabled_sources,
                "fps_cap": cfg.capture.fps_cap,
                "resolution": cfg.capture.resolution,
                "screen_monitor": cfg.capture.screen_monitor,
                "browser_scrub_pii": cfg.capture.browser_scrub_pii,
                "browser_exclude_patterns": cfg.capture.browser_exclude_patterns,
            }
            raw["privacy"] = {
                "exclude_apps": cfg.privacy.exclude_apps,
                "blur_regions": cfg.privacy.blur_regions,
                "scrub_pii": cfg.privacy.scrub_pii,
            }

            with open(path, "w") as f:
                yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

            log.info("Saved config to %s", config_path)
        except Exception as e:
            log.warning("Failed to save config: %s", e)
