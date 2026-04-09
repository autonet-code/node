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

            # Initialize blob store for weight delta uploads
            self._init_blob_store()

            logger.info(
                f"AutonetService initialized: "
                f"device={self.config.device}, "
                f"arch={self.config.model.architecture}, "
                f"training_feed={'active' if self._training_feed else 'disabled'}"
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
        """Upload weight delta to blob store and record training on RPB.

        This bridges the autonomous training loop to the network:
        1. Package delta + metrics into a model update
        2. Upload to blob store (content-addressed)
        3. Record the training contribution on RPB (mints future rewards)

        Returns the blob store CID if upload succeeded, None otherwise.
        """
        # Step 1: Upload to blob store
        delta_cid = None
        if self._blob_store:
            try:
                update = {
                    "weight_delta": {k: v.tolist() if hasattr(v, "tolist") else v
                                     for k, v in weight_delta.items()},
                    "metrics": metrics,
                    "source": "autonomous",
                }
                delta_cid = self._blob_store.add_json(update)
                if delta_cid:
                    logger.info("Weight delta uploaded to blob store: %s", delta_cid[:16])
            except Exception as e:
                logger.warning("Failed to upload delta to blob store: %s", e)

        # Step 2: Record training on-chain via RPB
        governance = getattr(self, "_governance", None)
        if governance and delta_cid:
            try:
                registry = governance.registry
                contribution = metrics.get("num_batches", 1)
                account = registry.blockchain.account
                if account:
                    result = registry.send("RPB", "recordTraining", account.address, contribution)
                    if result.success:
                        logger.info(
                            "Training recorded on-chain: contribution=%d, delta=%s",
                            contribution, delta_cid[:16],
                        )
                    else:
                        logger.warning("On-chain recordTraining failed: %s", result.error)
            except Exception as e:
                logger.warning("Failed to record training on-chain: %s", e)

        return delta_cid

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

    def notify_execution(self, agent_id: str, execution_id: str, status: str) -> None:
        """Notify the training feed that an agent execution completed.

        Called by AutonetBridge when EXECUTION_COMPLETED events fire.
        Thread-safe: the training feed's counter is incremented here
        (from the async event loop thread), and read in _run_cycle()
        (from the service's blocking thread).
        """
        if self._training_feed:
            self._training_feed.notify_execution(agent_id, execution_id, status)

    @staticmethod
    def _get_version() -> str:
        """Get the current node software version."""
        try:
            from nodes import __version__
            return __version__
        except (ImportError, AttributeError):
            return "0.1.0"
