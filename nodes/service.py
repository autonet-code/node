"""
Autonet Node Service — Managed Daemon Process.

Top-level entry point that runs a solver node as a managed background service.
Handles signal-based lifecycle (SIGTERM/SIGINT), resource monitoring,
auto-updates, and graceful shutdown.

Story 6.1: Solver node as daemon background service.
"""

import logging
import signal
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .common.config import AutonetConfig, load_config
from .common.resource_monitor import ResourceMonitor
from .common.updater import AutonetUpdater

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class ServiceStatus:
    """Snapshot of the service's current state."""
    state: ServiceState
    uptime_seconds: float = 0.0
    cycles_completed: int = 0
    errors: int = 0
    paused_reason: Optional[str] = None
    last_update_check: float = 0.0


class AutonetService:
    """
    Managed service wrapper around an Autonet solver node.

    Lifecycle:
        1. start() — initialize subsystems, enter main loop
        2. Main loop: cycle training, check resources, check updates
        3. stop() — graceful shutdown of all subsystems

    Signal handling:
        SIGTERM/SIGINT → stop()
    """

    def __init__(self, config: Optional[AutonetConfig] = None, data_dir: Optional[str] = None):
        self.config = config or load_config()
        self._state = ServiceState.STOPPED
        self._start_time: float = 0.0
        self._cycles: int = 0
        self._errors: int = 0
        self._shutdown_requested = False

        # Subsystems (lazy-init in start())
        self._resource_monitor: Optional[ResourceMonitor] = None
        self._updater: Optional[AutonetUpdater] = None
        self._node = None  # Solver or other node type

        # Training data feed (Story 3.2) — bridges agent activity → JEPA training
        self._training_feed = None
        self._data_dir = data_dir  # ATN data dir (~/.atn), passed from AutonetBridge

        # World-model substrate (Phase 2 of native integration).
        # Long-lived persistent World fed by agent activity. Lives at the
        # daemon level, one per RPB.
        self._world_service = None
        self._world_substrate_feed = None
        # Identity resolver for the substrate feed. Set before start()
        # so it's installed when _init_world_substrate_feed runs.
        # Signature: Callable[[str], Optional[ResolvedAgentIdentity]].
        self._substrate_identity_resolver: Any = None
        # Event gossip for substrate federation across daemons.
        # Constructed by attach_event_gossip() after the libp2p host
        # is ready (Phase 10.2).
        self._event_gossip = None
        # Federated-close driver: runs canonical_order +
        # federated_epoch_close on every epoch close, and decides
        # whether this daemon is the chosen on-chain submitter for
        # that epoch (Phase 10.3a). Subscribers fed the result land
        # in self._federated_close_subscribers.
        self._federated_close_driver = None
        self._federated_close_subscribers: list = []
        # Chain submission driver (Phase 10.3b/c): attached by
        # attach_chain_submission() after the federated-close driver
        # is up. Owns the EpochAnchorer (one) and per-agent
        # AuthoritativeChainSubmitter cache.
        self._chain_submission_driver = None
        # Phase 3: epoch scheduler ticks the World's reward windows.
        self._epoch_scheduler = None
        # Phase 4: external subscribers receive epoch close records.
        # Installed before start() so they're attached when the
        # WorldService is constructed.
        self._external_epoch_subscribers: list = []

        # Blob store for uploading weight deltas
        self._blob_store = None

    @property
    def state(self) -> ServiceState:
        return self._state

    def status(self) -> ServiceStatus:
        """Get current service status."""
        uptime = time.time() - self._start_time if self._start_time else 0.0
        return ServiceStatus(
            state=self._state,
            uptime_seconds=uptime,
            cycles_completed=self._cycles,
            errors=self._errors,
            paused_reason=(
                self._resource_monitor.paused_reason
                if self._resource_monitor else None
            ),
            last_update_check=self._updater._last_check if self._updater else 0.0,
        )

    def start(self):
        """
        Initialize subsystems and enter the main loop.

        This blocks until stop() is called (via signal or explicit call).
        """
        if self._state != ServiceState.STOPPED:
            logger.warning(f"Cannot start: service is {self._state.value}")
            return

        self._state = ServiceState.STARTING
        self._start_time = time.time()
        self._shutdown_requested = False

        # Configure logging
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

        logger.info("AutonetService starting...")

        # Install signal handlers
        self._install_signal_handlers()

        # Initialize subsystems
        try:
            self._resource_monitor = ResourceMonitor(self.config)
            self._updater = AutonetUpdater(
                config=self.config,
                current_version=self._get_version(),
            )
            # Pre-set last-check times so the first cycle doesn't trigger
            # expensive git-fetch / psutil calls immediately.
            now = time.time()
            self._resource_monitor._last_check = now
            self._updater._last_check = now

            # Initialize training data feed if ATN data dir is available
            if self._data_dir:
                self._init_training_feed()

            # Initialize world-model substrate (always available; feeding
            # is gated by config and data dir presence).
            self._init_world_service()
            if self._world_service is not None:
                self._init_epoch_scheduler()
            if self._data_dir and self._world_service is not None:
                self._init_world_substrate_feed()

            # Initialize blob store for weight delta uploads
            self._init_blob_store()

            logger.info(
                f"AutonetService initialized: "
                f"device={self.config.device}, "
                f"arch={self.config.model.architecture}, "
                f"training_feed={'active' if self._training_feed else 'disabled'}, "
                f"world_substrate={'active' if self._world_substrate_feed else 'disabled'}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize subsystems: {e}")
            self._state = ServiceState.STOPPED
            self._errors += 1
            return

        self._state = ServiceState.RUNNING
        logger.info("AutonetService running")

        # Main loop
        self._main_loop()

    def stop(self):
        """Request graceful shutdown."""
        if self._state in (ServiceState.STOPPED, ServiceState.STOPPING):
            return

        logger.info("AutonetService stopping...")
        self._state = ServiceState.STOPPING
        self._shutdown_requested = True

        # Stop the solver node if running
        if self._node and hasattr(self._node, "stop"):
            try:
                self._node.stop()
            except Exception as e:
                logger.error(f"Error stopping node: {e}")

        # Force a final epoch close so any in-flight reward window
        # is finalized rather than dropped silently.
        if self._epoch_scheduler is not None:
            try:
                self._epoch_scheduler.force_tick()
            except Exception as e:
                logger.error(f"Error in final epoch tick: {e}")

        # Flush + close WorldService so its persistent log is durable.
        if self._world_service is not None:
            try:
                # Close any epoch the scheduler didn't reach.
                if self._world_service.current_epoch_id is not None:
                    self._world_service.close_epoch()
                self._world_service.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down WorldService: {e}")

        self._state = ServiceState.STOPPED
        uptime = time.time() - self._start_time
        logger.info(
            f"AutonetService stopped after {uptime:.0f}s "
            f"({self._cycles} cycles, {self._errors} errors)"
        )

    def _main_loop(self):
        """Core service loop: train, monitor, update."""
        cycle_delay = self.config.node.cycle_delay
        max_cycles = self.config.node.max_cycles

        while not self._shutdown_requested:
            # Check cycle limit
            if max_cycles and self._cycles >= max_cycles:
                logger.info(f"Reached max_cycles ({max_cycles}), stopping")
                break

            try:
                # 1. Resource check
                if self._resource_monitor and not self._resource_monitor.should_train():
                    if self._state != ServiceState.PAUSED:
                        self._state = ServiceState.PAUSED
                        logger.info(
                            f"Training paused: {self._resource_monitor.paused_reason}"
                        )
                    time.sleep(cycle_delay)
                    continue
                elif self._state == ServiceState.PAUSED:
                    self._state = ServiceState.RUNNING
                    logger.info("Training resumed")

                # 2. Run one training cycle
                self._run_cycle()
                self._cycles += 1

                # 3. Check for updates (periodic)
                if self._updater and self._updater.should_check():
                    self._check_updates()

                # 4. Sleep between cycles
                time.sleep(cycle_delay)

            except Exception as e:
                logger.error(f"Error in service cycle {self._cycles}: {e}", exc_info=True)
                self._errors += 1
                time.sleep(cycle_delay)

        # Clean shutdown
        self.stop()

    def _run_cycle(self):
        """Execute one training/work cycle."""
        # If a solver node is attached, delegate to it
        if self._node and hasattr(self._node, "_run_cycle"):
            self._node._run_cycle()
            return

        # World-substrate feed: convert agent traces into substrate
        # events and feed them into the persistent World. Independent
        # of the JEPA path; both can run side-by-side until JEPA is
        # retired.
        if self._world_substrate_feed:
            try:
                substrate_metrics = self._world_substrate_feed.run_cycle()
                if substrate_metrics:
                    logger.info(
                        f"World-substrate cycle {self._cycles}: "
                        f"units={substrate_metrics.get('units_processed', 0)}, "
                        f"events={substrate_metrics.get('events_appended', 0)}"
                    )
            except Exception as e:
                logger.warning("World-substrate cycle failed: %s", e)

        # Epoch scheduler: tick to close+open reward windows on its
        # configured interval. Phase 4 will surface the close result
        # to the chain.
        if self._epoch_scheduler:
            try:
                self._epoch_scheduler.maybe_tick()
            except Exception as e:
                logger.warning("Epoch scheduler tick failed: %s", e)

        # Training data feed: train on accumulated agent activity
        if self._training_feed:
            result = self._training_feed.run_cycle()
            if result:
                weight_delta, metrics = result
                logger.info(
                    f"Training cycle {self._cycles}: "
                    f"loss={metrics.get('loss', 0):.4f}, "
                    f"batches={metrics.get('num_batches', 0)}"
                )

                # Upload delta to blob store and record on-chain
                delta_cid = self._submit_training_delta(weight_delta, metrics)

                # Update model state in P2P gossip so peers can surface stats
                self._update_gossip_model_state(metrics, delta_cid=delta_cid)
            return

        # Fallback: no node and no training feed
        logger.debug(f"Service cycle {self._cycles}")

    def _init_blob_store(self):
        """Initialize the blob store for weight delta uploads."""
        try:
            from .common.blob_store import BlobStore

            blob_dir = self.config.blob_store.data_dir if hasattr(self.config, "blob_store") else ""
            if not blob_dir:
                blob_dir = "~/.autonet/blobs"

            peer_urls = []
            if hasattr(self.config, "blob_store") and hasattr(self.config.blob_store, "peer_urls"):
                peer_urls = self.config.blob_store.peer_urls or []

            self._blob_store = BlobStore(data_dir=blob_dir, peer_urls=peer_urls)
            logger.info("Blob store initialized at %s", blob_dir)

        except Exception as e:
            logger.warning("Failed to initialize blob store: %s", e)
            self._blob_store = None

    def _submit_training_delta(self, weight_delta: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[str]:
        """Upload training contribution to blob store and record on RPB.

        Two contribution formats are supported:

          - JEPA path: weight_delta is a dict of named tensors. Uploaded
            as {"weight_delta": ...}; contribution = num_batches.
          - World-model substrate path: weight_delta carries the events
            stream and score snapshot. Uploaded as the contribution
            payload itself; contribution = previous-epoch mint amount
            looked up from the global model (or 1 if first epoch).

        Routing is by metrics["substrate"]: if "world-model", takes the
        substrate path; otherwise default to JEPA.
        """
        is_substrate = metrics.get("substrate") == "world-model"

        # Step 1: Upload to blob store
        delta_cid = None
        if self._blob_store:
            try:
                if is_substrate:
                    update = {
                        "events": weight_delta.get("events", []),
                        "score_snapshot_after": weight_delta.get(
                            "score_snapshot_after", {}),
                        "agent_id": weight_delta.get("agent_id", ""),
                        "metrics": metrics,
                        "substrate": "world-model",
                        "source": "autonomous",
                    }
                else:
                    update = {
                        "weight_delta": {k: v.tolist() if hasattr(v, "tolist") else v
                                         for k, v in weight_delta.items()},
                        "metrics": metrics,
                        "source": "autonomous",
                    }
                delta_cid = self._blob_store.add_json(update)
                if delta_cid:
                    logger.info("Training payload uploaded: %s (substrate=%s)",
                                delta_cid[:16], is_substrate)
            except Exception as e:
                logger.warning("Failed to upload payload to blob store: %s", e)

        # Step 2: Record training on-chain via RPB
        governance = getattr(self, "_governance", None)
        if governance and delta_cid:
            try:
                registry = governance.registry
                if is_substrate:
                    # Substrate path: contribution = mint earned in
                    # previous epoch (looked up from prior aggregation).
                    # Falls back to 1 on first epoch / lookup failure.
                    contribution = self._lookup_prior_mint() or 1
                else:
                    contribution = metrics.get("num_batches", 1)
                account = registry.blockchain.account
                if account:
                    result = registry.send("RPB", "recordTraining", contribution)
                    if result.success:
                        logger.info(
                            "Training recorded on-chain: contribution=%s, payload=%s",
                            contribution, delta_cid[:16],
                        )
                    else:
                        logger.warning("On-chain recordTraining failed: %s", result.error)
            except Exception as e:
                logger.warning("Failed to record training on-chain: %s", e)

        return delta_cid

    def _lookup_prior_mint(self) -> int:
        """Look up this agent's mint amount from the previous epoch's
        aggregated global model.

        The aggregator publishes serialized world payloads that include
        an `agent_mint` map. We fetch the latest mature model CID,
        download the payload, and read this agent's entry. Returns an
        integer (mint values are scaled to ints to match the contract's
        uint256 contribution field).

        Returns 0 if no prior model or no entry for this agent.
        """
        try:
            governance = getattr(self, "_governance", None)
            if not governance or not self._blob_store:
                return 0
            registry = governance.registry
            try:
                cid_result = registry.call("RPB", "getMatureModel")
                cid = cid_result[0] if isinstance(cid_result, (list, tuple)) else cid_result
            except Exception:
                return 0
            if not cid:
                return 0
            payload = self._blob_store.get_json(cid)
            if not payload:
                return 0
            agent_mint = payload.get("agent_mint", {}) or {}
            account = registry.blockchain.account
            if not account:
                return 0
            my_mint = float(agent_mint.get(str(account), 0.0))
            # Scale to integer units; mint values typically in [0, ~10]
            return max(1, int(my_mint * 1_000_000)) if my_mint > 0 else 0
        except Exception as e:
            logger.debug("prior-mint lookup failed: %s", e)
            return 0

    def _update_gossip_model_state(self, metrics: Dict[str, Any], delta_cid: Optional[str] = None) -> None:
        """Push training metrics + on-chain state into P2P gossip.

        Called after a training cycle completes.  Updates this node's
        ModelState in the NodeCapability, which gets broadcast to peers
        on the next advertise_capability() call.
        """
        try:
            from .common.p2p import AutonetHost
        except ImportError:
            return

        # Find the P2P host — it's managed by AutonetBridge, not us directly.
        # The service doesn't own the host, so we check if it was set externally.
        host = getattr(self, "_p2p_host", None)
        if not host or not isinstance(host, AutonetHost):
            return

        # Read on-chain state if governance bridge is available
        on_chain = {}
        governance = getattr(self, "_governance", None)
        if governance:
            try:
                rpb = governance.get_contract("RPB")
                if rpb:
                    on_chain["model_version"] = rpb.functions.modelVersion().call()
                    raw_hash = rpb.functions.currentModelHash().call()
                    on_chain["model_hash"] = "0x" + raw_hash.hex() if raw_hash else ""
                    on_chain["current_epoch"] = rpb.functions.currentEpoch().call()
                    on_chain["total_training_tokens"] = rpb.functions.totalTrainingTokens().call()
            except Exception as e:
                logger.debug("Failed to read on-chain model state: %s", e)

        kwargs = dict(
            model_version=on_chain.get("model_version", 0),
            model_hash=on_chain.get("model_hash", ""),
            current_epoch=on_chain.get("current_epoch", 0),
            total_training_tokens=on_chain.get("total_training_tokens", 0),
            local_cycles_completed=self._training_feed.cycles_completed if self._training_feed else 0,
            last_loss=metrics.get("loss", 0.0),
            last_cosine_similarity=metrics.get("cosine_similarity", 0.0),
            architecture=metrics.get("architecture", ""),
            param_count=metrics.get("param_count", 0),
        )
        if delta_cid:
            kwargs["latest_delta_cid"] = delta_cid
        host.update_model_state(**kwargs)
        logger.debug("Model state updated in P2P gossip")

    def _check_updates(self):
        """Check for and optionally apply updates."""
        try:
            info = self._updater.check_update()
            if info and info.has_update:
                logger.info(
                    f"Update available: {info.current_version} -> {info.available_version}"
                )
                if self.config.update.auto_apply:
                    logger.info("Auto-applying update...")
                    if self._updater.apply_update(info):
                        logger.info("Update applied. Restart recommended.")
                    else:
                        logger.error("Update apply failed")
        except Exception as e:
            logger.error(f"Update check error: {e}")

    def _install_signal_handlers(self):
        """Install SIGTERM and SIGINT handlers for graceful shutdown."""
        def _handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating shutdown...")
            self._shutdown_requested = True

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except (OSError, ValueError):
            # signal handlers can only be set in main thread
            logger.debug("Could not install signal handlers (not main thread?)")

    def _init_governance(self):
        """Initialize the governance bridge for on-chain attestation.

        Returns a GovernanceBridge if blockchain config has a private_key,
        or None for offline mode (training still works, just no attestation).
        """
        if not self.config.blockchain.private_key:
            logger.info("No blockchain private_key configured, governance offline")
            return None

        try:
            from .common.blockchain import BlockchainInterface
            from .common.contracts import ContractRegistry
            from .common.governance import GovernanceBridge

            blockchain = BlockchainInterface(
                rpc_url=self.config.blockchain.rpc_url,
                private_key=self.config.blockchain.private_key,
                chain_id=self.config.blockchain.chain_id,
            )

            if not blockchain.web3 or not blockchain.web3.is_connected():
                logger.warning("Blockchain not reachable, governance offline")
                return None

            registry = ContractRegistry(blockchain)

            governance = GovernanceBridge(
                registry=registry,
                node_id=f"service-{blockchain.account.address[:10]}" if blockchain.account else "service-anon",
            )

            logger.info(
                "Governance bridge initialized (rpc=%s, account=%s)",
                self.config.blockchain.rpc_url,
                blockchain.account.address if blockchain.account else "none",
            )
            return governance

        except Exception as e:
            logger.warning("Failed to initialize governance bridge: %s", e)
            return None

    def _init_training_feed(self):
        """Initialize the training data feed from ATN agent activity.

        Story 3.2: Reads conversations and execution records from the ATN
        data directory and feeds them into JEPA training.
        """
        try:
            from .common.training_feed import TrainingDataFeed, TrainingFeedConfig

            feed_config = TrainingFeedConfig(
                data_dir=self._data_dir,
                # Map from nodes config
                vocab_size=getattr(self.config.model, "vocab_size", 260),
                max_seq_length=getattr(self.config.model, "max_seq_length", 2048),
                embed_dim=getattr(self.config.model, "embed_dim", 64),
                num_heads=getattr(self.config.model, "num_heads", 4),
                encoder_depth=getattr(self.config.model, "encoder_depth", 2),
                predictor_depth=getattr(self.config.model, "predictor_depth", 1),
                predictor_embed_dim=getattr(self.config.model, "predictor_embed_dim", 32),
                epochs=self.config.training.epochs,
                learning_rate=self.config.training.learning_rate,
                batch_size=self.config.training.batch_size,
                backbone_model_id=getattr(self.config.model, "backbone_model_id", ""),
                scrub_pii=getattr(self.config, "privacy", None) is not None
                    and getattr(self.config.privacy, "scrub_pii", True),
            )

            self._governance = self._init_governance()
            self._training_feed = TrainingDataFeed(feed_config, governance=self._governance)
            logger.info(
                "Training data feed initialized (data_dir=%s, governance=%s)",
                self._data_dir,
                "online" if self._governance else "offline",
            )

        except Exception as e:
            logger.warning("Failed to initialize training data feed: %s", e)
            self._training_feed = None

    def _init_world_service(self):
        """Initialize the persistent World service (Phase 2 native
        integration). One ``WorldService`` per RPB address; lives at
        the daemon level for the lifetime of this process.

        The RPB address is currently derived from config (or governance
        when wired up); for now we use a stable per-host id so each
        daemon owns exactly one persistent world.
        """
        try:
            from .common.world_service import WorldService
            rpb_address = getattr(self.config, "rpb_address", None) \
                or getattr(self.config, "node_id", None) \
                or "default"
            self._world_service = WorldService(rpb_address=str(rpb_address))
            # Attach external subscribers registered before start().
            for handler in self._external_epoch_subscribers:
                self._world_service.subscribe_epoch_closed(handler)
            logger.info(
                "WorldService initialized for rpb=%s (embedding_dim=%d, %d subscribers)",
                rpb_address,
                self._world_service.embedding_dim,
                len(self._external_epoch_subscribers),
            )
        except Exception as e:
            logger.warning("Failed to initialize WorldService: %s", e)
            self._world_service = None

    def add_epoch_close_subscriber(self, handler) -> None:
        """Register a handler that will be invoked on every epoch
        close. Safe to call before start() — handlers attach to the
        WorldService when it's constructed in start()."""
        self._external_epoch_subscribers.append(handler)
        if self._world_service is not None:
            self._world_service.subscribe_epoch_closed(handler)

    def _init_epoch_scheduler(self):
        """Initialize the wall-clock epoch scheduler for reward windows.

        Phase 3 of native integration. The scheduler doesn't run its
        own thread by default — instead the daemon's main loop calls
        ``maybe_tick`` each cycle. Phase 4 will hand the close result
        off to a chain-reporting handler; for now we just log it.
        """
        try:
            from .common.epoch_scheduler import (
                EpochScheduler,
                EpochSchedulerConfig,
            )
            interval = float(getattr(self.config, "epoch_interval_seconds", 60.0))

            def _on_close(result: Dict[str, Any]) -> None:
                logger.info(
                    "Epoch closed: id=%s, n_events=%d, total_mint=%.4f, "
                    "agents=%d",
                    result.get("epoch_id"),
                    result.get("n_events", 0),
                    result.get("total_mint", 0.0),
                    len(result.get("agent_mint", {})),
                )

            self._epoch_scheduler = EpochScheduler(
                world_service=self._world_service,
                config=EpochSchedulerConfig(interval_seconds=interval),
                on_close=_on_close,
            )
            logger.info("Epoch scheduler initialized (interval=%.0fs)", interval)
        except Exception as e:
            logger.warning("Failed to initialize epoch scheduler: %s", e)
            self._epoch_scheduler = None

    def _init_world_substrate_feed(self):
        """Initialize the substrate feed that converts agent traces
        into substrate events bound for the persistent World."""
        try:
            from .common.world_substrate_feed import (
                WorldSubstrateFeed,
                WorldSubstrateFeedConfig,
            )
            feed_config = WorldSubstrateFeedConfig(
                data_dir=str(self._data_dir) if self._data_dir else "",
                agent_id=str(getattr(self.config, "node_id", "daemon")),
            )
            self._world_substrate_feed = WorldSubstrateFeed(
                config=feed_config,
                world_service=self._world_service,
                identity_resolver=self._substrate_identity_resolver,
            )
            logger.info(
                "World substrate feed initialized (data_dir=%s)",
                self._data_dir,
            )
        except Exception as e:
            logger.warning("Failed to initialize world substrate feed: %s", e)
            self._world_substrate_feed = None

    def notify_execution(self, agent_id: str, execution_id: str, status: str) -> None:
        """Notify the feeds that an agent execution completed.

        Called by AutonetBridge when EXECUTION_COMPLETED events fire.
        Thread-safe: the feeds' counters are incremented here (from the
        async event loop thread), and read in _run_cycle() (from the
        service's blocking thread).
        """
        if self._training_feed:
            self._training_feed.notify_execution(agent_id, execution_id, status)
        if self._world_substrate_feed:
            self._world_substrate_feed.notify_execution(agent_id, execution_id, status)

    def set_substrate_identity_resolver(self, resolver: Any) -> None:
        """Attach an identity resolver to the substrate feed.

        Resolver: ``Callable[[str], Optional[ResolvedAgentIdentity]]``.
        Called once per agent dir per cycle to map local agent_id ->
        on-chain identity. Agents without on-chain identity are
        skipped (their work has no network impact).

        Safe to call before or after start() — stored for the feed
        to pick up at construction, and forwarded to the feed if it
        already exists.
        """
        self._substrate_identity_resolver = resolver
        if self._world_substrate_feed is not None:
            self._world_substrate_feed.set_identity_resolver(resolver)

    def attach_event_gossip(self, p2p_host: Any) -> bool:
        """Wire the substrate event-gossip transport over libp2p.

        Caller is responsible for waiting until ``p2p_host._ready_event``
        is set (the host's trio loop is up and accepting publishes).
        This method then constructs an ``EventGossip`` per RPB and
        binds it to the WorldService's local-event stream.

        Returns True on success, False otherwise.
        """
        if self._world_service is None:
            logger.debug("attach_event_gossip: WorldService not yet initialized")
            return False
        if p2p_host is None:
            logger.debug("attach_event_gossip: no p2p_host")
            return False
        if self._event_gossip is not None:
            return True  # already wired
        try:
            from .common.event_gossip import (
                EventGossip,
                Keypair,
                LibP2PTransport,
            )
        except Exception as e:
            logger.warning("attach_event_gossip: import failed: %s", e)
            return False

        try:
            keypair = self._load_or_create_gossip_keypair()
            transport = LibP2PTransport(p2p_host)
            self._event_gossip = EventGossip(
                rpb_address=str(self._world_service.rpb_address),
                keypair=keypair,
                transport=transport,
                world_service=self._world_service,
            )
            logger.info(
                "Event gossip attached for rpb=%s (sender_pubkey=%s...)",
                self._world_service.rpb_address,
                keypair.public_key.hex()[:16],
            )
            self._init_federated_close_driver()
            return True
        except Exception as e:
            logger.warning("attach_event_gossip: %s", e, exc_info=True)
            self._event_gossip = None
            return False

    def _init_federated_close_driver(self) -> None:
        """Construct the federated-close driver and subscribe it to
        WorldService epoch-close events.

        Runs after EventGossip is attached. Each local epoch close
        triggers ``driver.run(local_close_result)`` which builds the
        canonical order from gossiped batches, runs
        ``federated_epoch_close``, and produces a deterministic
        authoritative result. Subscribers (chain submitter, etc.)
        receive the FederatedCloseResult.
        """
        if self._world_service is None or self._event_gossip is None:
            return
        try:
            from .common.federated_close_driver import FederatedCloseDriver
        except Exception as e:
            logger.warning("federated_close_driver import failed: %s", e)
            return
        embedding_dim = getattr(self._world_service, "embedding_dim", 1024)
        self._federated_close_driver = FederatedCloseDriver(
            gossip=self._event_gossip,
            embedding_dim=embedding_dim,
        )

        def _on_local_close(local_result):
            try:
                fed = self._federated_close_driver.run(local_result)
            except Exception as e:
                logger.error(
                    "federated close driver failed: %s", e, exc_info=True,
                )
                return
            if fed is None:
                return
            for sub in list(self._federated_close_subscribers):
                try:
                    sub(fed)
                except Exception as e:
                    logger.error(
                        "federated-close subscriber failed: %s", e, exc_info=True,
                    )

        self._world_service.subscribe_epoch_closed(_on_local_close)
        logger.info("Federated-close driver wired into WorldService")

    def add_federated_close_subscriber(self, handler: Any) -> None:
        """Register a callable invoked with the FederatedCloseResult
        each time a federated close completes. Used by chain wiring
        (Phase 10.3b/c) to anchor + submit per-agent mints.
        """
        self._federated_close_subscribers.append(handler)

    def attach_chain_submission(
        self,
        agent_chain_resolver: Any,
        *,
        blob_resolver: Any = None,
    ) -> bool:
        """Wire on-chain anchor + per-agent mint submission.

        Requires Phase 10.3a (federated-close driver) to be wired
        first. Reads chain config from
        ``self.config.blockchain.substrate_address`` etc.; if any
        required field is empty, this is a no-op.

        ``agent_chain_resolver`` is a zero-arg callable returning a
        list of ``AgentChainIdentity(address, private_key)`` for every
        currently-registered agent on this daemon. The atn-side
        autonet_service walks ``agent_registry.list_agents()`` to
        produce this.

        Returns True when chain submission is now active.
        """
        if self._federated_close_driver is None:
            logger.warning(
                "attach_chain_submission: federated-close driver not wired"
            )
            return False
        try:
            from .common.chain_submission_driver import (
                ChainSubmissionConfig,
                ChainSubmissionDriver,
            )
        except Exception as e:
            logger.warning("chain_submission_driver import failed: %s", e)
            return False

        bc = self.config.blockchain
        cfg = ChainSubmissionConfig(
            substrate_address=getattr(bc, "substrate_address", "") or "",
            rpc_url=bc.rpc_url,
            chain_id=bc.chain_id,
            daemon_private_key=bc.private_key or "",
        )
        try:
            driver = ChainSubmissionDriver(
                config=cfg,
                agent_chain_resolver=agent_chain_resolver,
                blob_resolver=blob_resolver,
            )
        except Exception as e:
            logger.warning("ChainSubmissionDriver build failed: %s", e)
            return False
        if not driver.enabled:
            logger.info("Chain submission disabled (no substrate_address or key)")
            return False

        self._chain_submission_driver = driver
        self.add_federated_close_subscriber(driver.handle_federated_close)
        logger.info(
            "Chain submission attached (substrate=%s)", cfg.substrate_address,
        )
        return True

    def _load_or_create_gossip_keypair(self):
        """Load this daemon's persistent ed25519 gossip keypair, or
        create one and persist it under the data dir.

        The key is per-daemon (not per-agent) — it identifies the
        gossip-batch sender, separate from any on-chain agent identity.
        """
        from .common.event_gossip import Keypair
        import json
        if not self._data_dir:
            return Keypair.generate()
        key_dir = self._data_dir / "world" / "gossip"
        key_dir.mkdir(parents=True, exist_ok=True)
        key_path = key_dir / "keypair.json"
        if key_path.exists():
            try:
                data = json.loads(key_path.read_text(encoding="utf-8"))
                return Keypair(
                    public_key=bytes.fromhex(data["public_key"]),
                    private_key=bytes.fromhex(data["private_key"]),
                )
            except Exception as e:
                logger.warning("gossip keypair load failed (regenerating): %s", e)
        kp = Keypair.generate()
        key_path.write_text(
            json.dumps({
                "public_key": kp.public_key.hex(),
                "private_key": kp.private_key.hex(),
            }),
            encoding="utf-8",
        )
        return kp

    @staticmethod
    def _get_version() -> str:
        """Get the current node software version."""
        try:
            from nodes import __version__
            return __version__
        except (ImportError, AttributeError):
            return "0.1.0"
