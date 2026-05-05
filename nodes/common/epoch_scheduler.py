"""Wall-clock scheduler for WorldService epoch boundaries.

Phase 3 of native world-model integration. Ticks at a fixed wall-clock
interval; on each tick, closes the current epoch and opens the next.
The receipt from ``close_epoch`` (per-agent mint, novelty, etc) is
delivered to a caller-supplied handler — Phase 4 will wire that handler
to chain reporting.

Two ways to run
---------------

1. **Inline / on-demand**: call ``maybe_tick()`` from the daemon's
   main loop. Works without threads. Suited to integration with
   ``AutonetService._run_cycle``.

2. **Background thread**: call ``start()`` to spin a daemon thread
   that ticks autonomously. Suited to a daemon that doesn't have a
   long-running cycle of its own.

Phase 3 wires (1) into ``AutonetService``. (2) is provided for tests
and standalone tooling.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


# Default interval tuned for sim runs (60s) — production should bump
# this via config; see plan's "epoch length" decision.
_DEFAULT_INTERVAL_SECONDS = 60.0


# A handler takes the result of WorldService.close_epoch and does
# something with it (chain reporting in Phase 4, in-memory recording
# for tests, etc).
EpochResultHandler = Callable[[Dict[str, Any]], None]


@dataclass
class EpochSchedulerConfig:
    interval_seconds: float = _DEFAULT_INTERVAL_SECONDS
    apply_gate: bool = True
    gate_strength: float = 1.0
    # If True, the scheduler opens the first epoch on construction so
    # the daemon starts buffering events immediately.
    open_first_epoch_on_start: bool = True


class EpochScheduler:
    """Drives ``WorldService.open_epoch`` / ``close_epoch`` on a clock."""

    def __init__(
        self,
        world_service: Any,
        config: Optional[EpochSchedulerConfig] = None,
        on_close: Optional[EpochResultHandler] = None,
    ):
        if world_service is None:
            raise ValueError("EpochScheduler requires a world_service")
        self.world_service = world_service
        self.config = config or EpochSchedulerConfig()
        self._on_close = on_close
        self._last_open_at: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._n_closed = 0

        if self.config.open_first_epoch_on_start:
            self._open_epoch()

    # ------------------------------------------------------------------
    # Tick semantics
    # ------------------------------------------------------------------

    def maybe_tick(self) -> Optional[Dict[str, Any]]:
        """Close+reopen if the interval has elapsed. No-op otherwise.

        Returns the close result if a tick happened, else None.
        """
        if self._last_open_at == 0.0:
            self._open_epoch()
            return None
        elapsed = time.time() - self._last_open_at
        if elapsed < self.config.interval_seconds:
            return None
        return self._tick()

    def force_tick(self) -> Dict[str, Any]:
        """Close the current epoch immediately and open the next.
        Useful for tests and end-of-shutdown handling."""
        return self._tick()

    @property
    def closed_count(self) -> int:
        return self._n_closed

    # ------------------------------------------------------------------
    # Background thread mode
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="epoch-scheduler", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.maybe_tick()
            except Exception as e:
                logger.error("epoch scheduler tick failed: %s", e, exc_info=True)
            # Wait either the configured interval, or until stop.
            self._stop_event.wait(min(self.config.interval_seconds, 5.0))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_epoch(self) -> None:
        self.world_service.open_epoch()
        self._last_open_at = time.time()

    def _tick(self) -> Dict[str, Any]:
        result = self.world_service.close_epoch(
            apply_gate=self.config.apply_gate,
            gate_strength=self.config.gate_strength,
        )
        self._n_closed += 1
        if self._on_close is not None:
            try:
                self._on_close(result)
            except Exception as e:
                logger.error(
                    "epoch close handler failed: %s", e, exc_info=True,
                )
        # Open the next epoch.
        self._open_epoch()
        return result
