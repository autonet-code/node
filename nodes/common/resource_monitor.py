"""
Resource Monitor for Autonet Nodes.

Tracks CPU, memory, and GPU usage. Enforces configurable limits
and active-hours scheduling to avoid overwhelming the host machine.

Story 6.2: Resource limit configuration.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .config import AutonetConfig, ResourceConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Point-in-time resource usage reading.

    ``memory_mb`` is THIS PROCESS's RSS — the thing the configured
    ``max_memory_mb`` budget bounds. ``system_available_mb`` is the
    host's free headroom, used as a pressure floor. (Comparing total
    system-used memory against the budget — the old behavior — paused
    training forever on any desktop where other apps held >4GB.)
    """
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    system_available_mb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    timestamp: float = 0.0


class ResourceMonitor:
    """
    Monitors system resources and decides whether training should proceed.

    Usage:
        monitor = ResourceMonitor(config)
        if monitor.should_train():
            # proceed with training
        snapshot = monitor.snapshot()
    """

    def __init__(self, config: Optional[AutonetConfig] = None):
        self.config = config or load_config()
        self.resource_config: ResourceConfig = self.config.resources
        self._last_snapshot: Optional[ResourceSnapshot] = None
        self._last_check: float = 0.0
        self._paused_reason: Optional[str] = None

    def snapshot(self) -> ResourceSnapshot:
        """
        Take a point-in-time reading of system resources.

        Uses psutil for CPU/memory. GPU is best-effort via torch.cuda.
        """
        snap = ResourceSnapshot(timestamp=time.time())

        try:
            import psutil
            snap.cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            snap.memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            snap.memory_percent = mem.percent
            snap.system_available_mb = mem.available / (1024 * 1024)
        except ImportError:
            logger.debug("psutil not available, CPU/memory monitoring disabled")
        except Exception as e:
            logger.debug(f"Resource snapshot error: {e}")

        # GPU metrics (best-effort)
        try:
            import torch
            if torch.cuda.is_available():
                snap.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                # torch doesn't expose GPU utilization directly;
                # would need nvidia-ml-py for that. Leave at 0.
        except ImportError:
            pass
        except Exception:
            pass

        self._last_snapshot = snap
        return snap

    def should_train(self) -> bool:
        """
        Check whether training should proceed given current resource usage
        and active-hours schedule.

        Returns True if all limits are satisfied and we're within active hours.
        """
        # Rate-limit checks
        now = time.time()
        if now - self._last_check < self.resource_config.check_interval:
            # Re-use last decision if check interval hasn't elapsed
            return self._paused_reason is None

        self._last_check = now
        snap = self.snapshot()

        # Check active hours
        if not self._within_active_hours():
            self._paused_reason = "outside active hours"
            logger.debug(f"Training paused: {self._paused_reason}")
            return False

        # Check CPU
        if snap.cpu_percent > self.resource_config.max_cpu_percent:
            self._paused_reason = f"CPU {snap.cpu_percent:.1f}% > {self.resource_config.max_cpu_percent}%"
            logger.debug(f"Training paused: {self._paused_reason}")
            return False

        # Check memory: the budget bounds OUR process; a system-pressure
        # floor protects the host overall (and self-recovers when other
        # apps release RAM).
        if snap.memory_mb > self.resource_config.max_memory_mb:
            self._paused_reason = (
                f"Daemon memory {snap.memory_mb:.0f}MB > "
                f"{self.resource_config.max_memory_mb}MB"
            )
            logger.debug(f"Training paused: {self._paused_reason}")
            return False
        min_free_mb = float(getattr(self.resource_config, "min_free_memory_mb", 512))
        if snap.system_available_mb and snap.system_available_mb < min_free_mb:
            self._paused_reason = (
                f"System memory low ({snap.system_available_mb:.0f}MB free "
                f"< {min_free_mb:.0f}MB)"
            )
            logger.debug(f"Training paused: {self._paused_reason}")
            return False

        # All clear
        if self._paused_reason:
            logger.info(f"Training resumed (was paused: {self._paused_reason})")
        self._paused_reason = None
        return True

    def _within_active_hours(self) -> bool:
        """Check if current time is within configured active hours."""
        start = self.resource_config.active_hours_start
        end = self.resource_config.active_hours_end

        # 0-24 means always active
        if start == 0 and end == 24:
            return True

        current_hour = datetime.now().hour

        if start < end:
            # Normal range: e.g. 9-17
            return start <= current_hour < end
        else:
            # Overnight range: e.g. 22-6
            return current_hour >= start or current_hour < end

    @property
    def paused_reason(self) -> Optional[str]:
        """Returns the reason training is paused, or None if not paused."""
        return self._paused_reason

    @property
    def last_snapshot(self) -> Optional[ResourceSnapshot]:
        return self._last_snapshot
