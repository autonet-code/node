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
from typing import Optional

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

    def __init__(self, config: Optional[AutonetConfig] = None):
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

            logger.info(
                f"AutonetService initialized: "
                f"device={self.config.device}, "
                f"arch={self.config.model.architecture}"
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

        # Placeholder: in production, the node is created during start()
        # based on the configured role. For now, log the cycle.
        logger.debug(f"Service cycle {self._cycles}")

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

    @staticmethod
    def _get_version() -> str:
        """Get the current node software version."""
        try:
            from nodes import __version__
            return __version__
        except (ImportError, AttributeError):
            return "0.1.0"
