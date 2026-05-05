"""World-substrate feed — bridges agent activity to the persistent World.

Substrate-side counterpart to ``training_feed.TrainingDataFeed``.

Where the JEPA training feed reads agent JSONL traces and produces
weight-tensor deltas, this feed reads the same traces and produces
**substrate events** that flow into the daemon's ``WorldService``.

Data flow
---------

  Runtime EventBus
    -> EXECUTION_COMPLETED event
    -> WorldSubstrateFeed.notify_execution()
    -> Increments pending_events counter
  AutonetService._main_loop()
    -> WorldSubstrateFeed.run_cycle()
    -> Walks ~/.atn/agents/*/conversations/*.jsonl
    -> Builds (problem, resolution, outcome) tuples for new conversations
    -> Calls WorldService.submit_work_units(...)
    -> Returns metrics dict

Phase 2 of native world-model integration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WorldSubstrateFeedConfig:
    """Configuration for the substrate feed."""
    data_dir: str = ""
    min_events_for_cycle: int = 5
    min_cycle_interval: float = 60.0
    # Maximum work units processed per cycle (prevents pathological
    # bursts when the trace dir has thousands of unprocessed sessions).
    max_units_per_cycle: int = 32
    # Agent id used as ``author_agent`` on emitted events. Defaults to
    # this daemon's primary agent identity.
    agent_id: str = "daemon"


class WorldSubstrateFeed:
    """Drives substrate event ingestion from agent traces.

    Lifecycle mirrors TrainingDataFeed:
      1. Created by AutonetService when configured.
      2. notify_execution() called when agent executions complete.
      3. run_cycle() called periodically by the service main loop.
      4. Internally: walks conversations, builds work units, submits
         to WorldService.
    """

    def __init__(
        self,
        config: WorldSubstrateFeedConfig,
        world_service: Any,
    ):
        if world_service is None:
            raise ValueError("WorldSubstrateFeed requires a world_service")
        self.config = config
        self.world_service = world_service
        self._data_dir = Path(config.data_dir) if config.data_dir else None

        self._pending_events: int = 0
        self._total_events: int = 0
        self._cycles_completed: int = 0
        self._last_cycle_time: float = 0.0
        self._last_metrics: Optional[Dict[str, Any]] = None

        # Conversations we've already converted to work units. Keyed by
        # (agent, conversation_filename). Survives only while the
        # daemon process lives — restart re-processes everything (the
        # WorldService dedupes via content-addressed obs ids, so this
        # is safe but wasteful).
        self._processed: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    @property
    def pending_events(self) -> int:
        return self._pending_events

    @property
    def cycles_completed(self) -> int:
        return self._cycles_completed

    @property
    def last_metrics(self) -> Optional[Dict[str, Any]]:
        return self._last_metrics

    def notify_execution(self, agent_id: str, execution_id: str, status: str) -> None:
        """Bumps the pending-events counter when an execution completes."""
        if status not in ("completed", "failed"):
            return
        self._pending_events += 1
        self._total_events += 1

    def should_run(self) -> bool:
        if not self._data_dir:
            return False
        if self._pending_events < self.config.min_events_for_cycle:
            return False
        elapsed = time.time() - self._last_cycle_time
        if elapsed < self.config.min_cycle_interval:
            return False
        return True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_cycle(self) -> Optional[Dict[str, Any]]:
        """Run one substrate-feed cycle if conditions are met.

        Returns the cycle's metrics dict, or None if skipped.
        """
        if not self.should_run():
            return None

        logger.info(
            "Substrate feed: starting cycle %d (%d pending, %d total events)",
            self._cycles_completed + 1,
            self._pending_events,
            self._total_events,
        )

        try:
            metrics = self._do_cycle()
            self._pending_events = 0
            self._last_cycle_time = time.time()
            self._cycles_completed += 1
            self._last_metrics = metrics
            return metrics
        except Exception as e:
            logger.error("Substrate feed: cycle failed: %s", e, exc_info=True)
            self._pending_events = 0
            self._last_cycle_time = time.time()
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_cycle(self) -> Dict[str, Any]:
        """Walk the ATN data dir, build work units, submit to world."""
        from .world_model_substrate.outcomes import outcome_from_conversation

        start = time.time()
        work_units = self._collect_work_units()
        if not work_units:
            return {
                "units_processed": 0,
                "elapsed_seconds": time.time() - start,
                "skipped_reason": "no_new_work_units",
            }

        # Cap to avoid pathological bursts.
        if len(work_units) > self.config.max_units_per_cycle:
            work_units = work_units[: self.config.max_units_per_cycle]

        receipt = self.world_service.submit_work_units(
            work_units=work_units,
            agent_id=self.config.agent_id,
        )

        elapsed = time.time() - start
        metrics = {
            "units_processed": receipt.get("units_processed", 0),
            "events_appended": receipt.get("events_appended", 0),
            "rounds": receipt.get("rounds", 0),
            "elapsed_seconds": elapsed,
            "root_scores": receipt.get("root_scores_after", {}),
        }
        logger.info(
            "Substrate feed: %d work units, %d events, %.2fs",
            metrics["units_processed"],
            metrics["events_appended"],
            elapsed,
        )
        return metrics

    def _collect_work_units(self) -> List[Tuple[str, str, Any]]:
        """Walk agent dirs, build (problem, resolution, outcome) tuples
        for any conversations not yet processed in this daemon
        lifetime."""
        from .world_model_substrate.outcomes import (
            outcome_from_conversation,
            read_conversation,
        )

        if not self._data_dir or not self._data_dir.exists():
            return []

        agents_dir = self._data_dir / "agents"
        if not agents_dir.is_dir():
            return []

        units: List[Tuple[str, str, Any]] = []
        for agent_dir in sorted(agents_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            conv_dir = agent_dir / "conversations"
            if not conv_dir.is_dir():
                continue
            files = sorted(conv_dir.glob("*.jsonl"))
            for i, path in enumerate(files):
                key = (agent_dir.name, path.name)
                if key in self._processed:
                    continue
                messages = read_conversation(path)
                if not messages:
                    self._processed.add(key)
                    continue
                problem = self._extract_problem(messages)
                resolution = self._extract_resolution(messages)
                if not problem or not resolution:
                    self._processed.add(key)
                    continue
                later = files[i + 1:]
                outcome = outcome_from_conversation(path, later)
                units.append((problem, resolution, outcome))
                self._processed.add(key)
        return units

    @staticmethod
    def _extract_problem(messages: List[Dict[str, Any]]) -> str:
        """First user message is the problem. Concatenate adjacent user
        messages at the start in case they were split across turns."""
        out: list[str] = []
        for msg in messages:
            if msg.get("role") != "user":
                if out:
                    break
                continue
            text = msg.get("content", "") or ""
            if isinstance(text, list):
                # Multi-part content blocks
                text = " ".join(
                    p.get("text", "") for p in text if isinstance(p, dict)
                )
            if text:
                out.append(text)
        return "\n".join(out).strip()

    @staticmethod
    def _extract_resolution(messages: List[Dict[str, Any]]) -> str:
        """Last assistant message is the resolution."""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            text = msg.get("content", "") or ""
            if isinstance(text, list):
                text = " ".join(
                    p.get("text", "") for p in text if isinstance(p, dict)
                )
            text = (text or "").strip()
            if text:
                return text
        return ""
