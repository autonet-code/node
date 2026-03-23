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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

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
    # Training metrics
    cycles_completed: int = 0
    uptime_seconds: float = 0.0
    errors: int = 0
    last_error: str = ""
    # Resource usage (from ResourceMonitor)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_available: bool = False
    gpu_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "wallet_connected": self.wallet_connected,
            "wallet_address": self.wallet_address,
            "chain_id": self.chain_id,
            "rpc_url": self.rpc_url,
            "cycles_completed": self.cycles_completed,
            "uptime_seconds": self.uptime_seconds,
            "errors": self.errors,
            "last_error": self.last_error,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "gpu_available": self.gpu_available,
            "gpu_memory_mb": self.gpu_memory_mb,
        }


class AutonetBridge:
    """Bridge between ATN Runtime and the Autonet training service.

    Manages the lifecycle of the AutonetService and exposes control/status
    methods for the WS API.  Runs the training loop in a background asyncio
    task so it doesn't block the ATN event loop.
    """

    def __init__(self, config: AutonetConfig, event_bus: EventBus | None = None) -> None:
        self.config = config
        self.state = AutonetState()
        self._events = event_bus
        self._service = None      # Will hold nodes.service.AutonetService
        self._task: asyncio.Task | None = None
        self._autonet_config = None  # Will hold nodes.common.config.AutonetConfig

        # Apply config values to state
        if config.wallet_address:
            self.state.wallet_connected = True
            self.state.wallet_address = config.wallet_address
        if config.rpc_url:
            self.state.rpc_url = config.rpc_url
        if config.chain_id:
            self.state.chain_id = config.chain_id

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

            # Create and start the service in a background thread
            # (AutonetService.start() is blocking)
            self._service = AutonetService(self._autonet_config)
            self._task = asyncio.create_task(self._run_service())

            self.state.status = AutonetStatus.RUNNING
            log.info("Autonet service started")
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
        from nodes.common.config import load_config as load_autonet_config
        config_path = self.config.config_path or None
        return load_autonet_config(config_path)

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
