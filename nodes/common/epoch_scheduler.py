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
    # Candle close (anti last-second-spam; see docs/epoch_economics.md):
    # when candle_min_seconds > 0, an epoch runs the FULL
    # min+window duration, then closes at a cutoff drawn retroactively
    # inside the window — unknowable while the epoch is open. Events
    # after the cutoff roll into the next epoch. interval_seconds is
    # ignored in candle mode.
    candle_min_seconds: float = 0.0
    candle_window_seconds: float = 0.0


class EpochScheduler:
    """Drives ``WorldService.open_epoch`` / ``close_epoch`` on a clock."""

    def __init__(
        self,
        world_service: Any,
        config: Optional[EpochSchedulerConfig] = None,
        on_close: Optional[EpochResultHandler] = None,
        candle_seed_source: Optional[Callable[[float], Optional[bytes]]] = None,
    ):
        if world_service is None:
            raise ValueError("EpochScheduler requires a world_service")
        self.world_service = world_service
        self.config = config or EpochSchedulerConfig()
        self._on_close = on_close
        # Optional shared-randomness source for the candle cut (see
        # candle_seed.ChainCandleSeed). Called with the window's end
        # timestamp; returns 32 bytes every daemon agrees on, or None
        # to fall back to the local hash seed (single-daemon dev).
        self._candle_seed_source = candle_seed_source
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
        """Close+reopen if the epoch's time is up. No-op otherwise.

        Plain mode: closes after ``interval_seconds``. Candle mode
        (``candle_min_seconds > 0``): closes only after the FULL
        ``min + window`` has elapsed, at a cutoff drawn retroactively
        inside the window — so while the epoch is open, nobody can
        know which moment will turn out to have been the last.

        Returns the close result if a tick happened, else None.
        """
        if self._last_open_at == 0.0:
            self._open_epoch()
            return None
        elapsed = time.time() - self._last_open_at
        if self.config.candle_min_seconds > 0:
            full = self.config.candle_min_seconds + self.config.candle_window_seconds
            if elapsed < full:
                return None
            return self._tick(cutoff_ts=self._draw_candle_cut())
        if elapsed < self.config.interval_seconds:
            return None
        return self._tick()

    def force_tick(self) -> Dict[str, Any]:
        """Close the current epoch immediately and open the next.
        Useful for tests and end-of-shutdown handling."""
        return self._tick()

    def _draw_candle_cut(self) -> float:
        """Retroactively draw the effective cutoff inside the window.

        Seed preference order:

          1. ``candle_seed_source`` (chain-derived; see
             candle_seed.ChainCandleSeed): latestAnchorHash + the hash
             of the first block past the window's end. Shared across
             daemons AND unknowable while the epoch was open.
          2. Local fallback: hash of (previous epoch's identity, this
             epoch's id) — agreed by construction on a single daemon,
             unknowable before the window because the draw only
             happens at T_max. Fine for dev; NOT federation-safe.

        The cut function is source-agnostic: it just consumes 32 bytes.
        VRF remains the pre-mainnet upgrade path if block-producer
        grinding becomes a concern.
        """
        import hashlib
        current_id = str(getattr(self.world_service, "current_epoch_id", "") or "")

        seed: Optional[bytes] = None
        if self._candle_seed_source is not None:
            window_end = self._last_open_at + (
                self.config.candle_min_seconds + self.config.candle_window_seconds
            )
            try:
                seed = self._candle_seed_source(window_end)
            except Exception as e:
                logger.warning("candle seed source failed: %s", e)
            if seed is not None:
                logger.info("candle cut seeded from chain for epoch=%s", current_id)

        if seed is None:
            prev_id, prev_closed = "", 0.0
            try:
                history = self.world_service.epoch_history
                if history:
                    prev_id = str(history[-1].get("epoch_id", ""))
                    prev_closed = float(history[-1].get("closed_at", 0.0))
            except Exception:
                pass
            seed = hashlib.sha256(
                f"{prev_id}|{prev_closed}|{current_id}".encode("utf-8")
            ).digest()
        frac = int.from_bytes(seed[:8], "big") / float(2 ** 64)
        offset = self.config.candle_min_seconds + frac * self.config.candle_window_seconds
        t_cut = self._last_open_at + offset
        logger.info(
            "candle cut drawn: epoch=%s, offset=%.0fs into the window "
            "(min=%.0fs, window=%.0fs)",
            current_id, offset - self.config.candle_min_seconds,
            self.config.candle_min_seconds, self.config.candle_window_seconds,
        )
        return t_cut

    @property
    def closed_count(self) -> int:
        return self._n_closed

    def status(self) -> Dict[str, Any]:
        """Scheduler-side half of the epoch_status surface (the
        WorldService owns the other half). All times are unix seconds.

        ``t_max`` is when the close fires: in candle mode that's the
        moment the cutoff is DRAWN (the effective end lands earlier,
        somewhere inside the window — by design unknowable until then).
        """
        cfg = self.config
        mode = "candle" if cfg.candle_min_seconds > 0 else "interval"
        opened = self._last_open_at or None
        t_max = None
        if opened:
            if mode == "candle":
                t_max = opened + cfg.candle_min_seconds + cfg.candle_window_seconds
            else:
                t_max = opened + cfg.interval_seconds
        return {
            "mode": mode,
            "opened_at": opened,
            "t_max": t_max,
            "candle_min_seconds": cfg.candle_min_seconds if mode == "candle" else None,
            "candle_window_seconds": cfg.candle_window_seconds if mode == "candle" else None,
            "interval_seconds": cfg.interval_seconds if mode == "interval" else None,
            "seed_source": "chain" if self._candle_seed_source is not None else "local",
            "closed_count": self._n_closed,
        }

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

    def _tick(self, cutoff_ts: Optional[float] = None) -> Dict[str, Any]:
        result = self.world_service.close_epoch(
            apply_gate=self.config.apply_gate,
            gate_strength=self.config.gate_strength,
            cutoff_ts=cutoff_ts,
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
