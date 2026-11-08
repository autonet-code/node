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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ResolvedAgentIdentity:
    """The minimal identity shape the feed needs from the agent registry.

    Kept independent of atn.models.AgentIdentity so the nodes layer
    doesn't import atn.
    """
    address: str
    registered_on_chain: bool


# Resolver: local agent_id (YAML id) -> ResolvedAgentIdentity or None.
# None means "no on-chain identity"; the feed will skip work units
# from such agents because they have no impact on the network
# (no 0x address means no mint attribution).
IdentityResolver = Callable[[str], Optional[ResolvedAgentIdentity]]


@dataclass
class WorldSubstrateFeedConfig:
    """Configuration for the substrate feed."""
    data_dir: str = ""
    min_events_for_cycle: int = 5
    min_cycle_interval: float = 60.0
    # Maximum work units processed per cycle (prevents pathological
    # bursts when the trace dir has thousands of unprocessed sessions).
    max_units_per_cycle: int = 32
    # Fallback agent id used as ``author_agent`` only when no
    # identity_resolver is set (e.g. tests). Production daemons set
    # the resolver so each work unit is attributed to the actual
    # agent's 0x address.
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
        identity_resolver: Optional[IdentityResolver] = None,
    ):
        if world_service is None:
            raise ValueError("WorldSubstrateFeed requires a world_service")
        self.config = config
        self.world_service = world_service
        self._identity_resolver: Optional[IdentityResolver] = identity_resolver
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

    def set_identity_resolver(self, resolver: Optional[IdentityResolver]) -> None:
        """Late-bind the identity resolver. Daemons construct the feed
        before the agent registry is fully wired, so this lets the
        resolver be attached afterwards.
        """
        self._identity_resolver = resolver

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
        """Walk the ATN data dir, build per-agent work units, submit to world.

        Work units are grouped by the authoring agent's resolved 0x
        address. Agents without an on-chain identity (no resolver, or
        resolver returns None / registered_on_chain=False) are skipped:
        without a 0x address they have no impact on the network.
        """
        start = time.time()
        units_by_agent = self._collect_work_units_by_agent()
        if not units_by_agent:
            return {
                "units_processed": 0,
                "elapsed_seconds": time.time() - start,
                "skipped_reason": "no_new_work_units",
            }

        # Cap total units across all agents to avoid pathological bursts.
        cap = self.config.max_units_per_cycle
        total = sum(len(u) for u in units_by_agent.values())
        if total > cap:
            # Trim from the largest groups first so each agent keeps at
            # least some signal where possible.
            ordered = sorted(units_by_agent.items(), key=lambda kv: len(kv[1]), reverse=True)
            units_by_agent = {}
            remaining = cap
            for addr, units in ordered:
                if remaining <= 0:
                    break
                take = min(len(units), remaining)
                units_by_agent[addr] = units[:take]
                remaining -= take

        total_units_processed = 0
        total_events_appended = 0
        last_root_scores: Dict[str, Any] = {}
        for address, work_units in units_by_agent.items():
            receipt = self.world_service.submit_work_units(
                work_units=work_units,
                agent_id=address,
            )
            total_units_processed += int(receipt.get("units_processed", 0))
            total_events_appended += int(receipt.get("events_appended", 0))
            last_root_scores = receipt.get("root_scores_after", last_root_scores)

        elapsed = time.time() - start
        metrics = {
            "units_processed": total_units_processed,
            "events_appended": total_events_appended,
            "agents_attributed": len(units_by_agent),
            "elapsed_seconds": elapsed,
            "root_scores": last_root_scores,
        }
        logger.info(
            "Substrate feed: %d work units across %d agent(s), %d events, %.2fs",
            metrics["units_processed"],
            metrics["agents_attributed"],
            metrics["events_appended"],
            elapsed,
        )
        return metrics

    def _collect_work_units_by_agent(self) -> Dict[str, List[Tuple[str, str, Any]]]:
        """Walk agent dirs, build (problem, resolution, outcome) tuples
        for new conversations, grouped by the authoring agent's
        resolved 0x address.

        Agents without an on-chain identity are skipped entirely. When
        no identity_resolver is set (e.g. unit tests), all units fall
        back to ``self.config.agent_id`` so existing test code paths
        keep working.
        """
        from .world_model_substrate.outcomes import (
            outcome_from_conversation,
            read_conversation,
        )

        if not self._data_dir or not self._data_dir.exists():
            return {}

        agents_dir = self._data_dir / "agents"
        if not agents_dir.is_dir():
            return {}

        out: Dict[str, List[Tuple[str, str, Any]]] = defaultdict(list)
        for agent_dir in sorted(agents_dir.iterdir()):
            if not agent_dir.is_dir():
                continue

            local_id = agent_dir.name
            author = self._resolve_author(local_id)
            if author is None:
                # Skip — agent has no on-chain identity, so their work
                # has no network impact. Mark conversations processed
                # so we don't repeatedly re-walk them.
                self._mark_agent_dir_processed(agent_dir)
                continue

            conv_dir = agent_dir / "conversations"
            if not conv_dir.is_dir():
                continue
            files = sorted(conv_dir.glob("*.jsonl"))
            for i, path in enumerate(files):
                key = (local_id, path.name)
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
                out[author].append((problem, resolution, outcome))
                self._processed.add(key)
        return dict(out)

    # Backwards-compatible: tests still call _collect_work_units(); flatten.
    def _collect_work_units(self) -> List[Tuple[str, str, Any]]:
        grouped = self._collect_work_units_by_agent()
        flat: List[Tuple[str, str, Any]] = []
        for units in grouped.values():
            flat.extend(units)
        return flat

    def _resolve_author(self, local_agent_id: str) -> Optional[str]:
        """Return the 0x address to attribute work units to, or None
        if the agent has no on-chain identity (and should be skipped).

        With no resolver set, fall back to the configured default —
        this keeps tests working without forcing them to wire a
        registry.
        """
        if self._identity_resolver is None:
            return self.config.agent_id
        identity = self._identity_resolver(local_agent_id)
        if identity is None:
            return None
        if not identity.registered_on_chain or not identity.address:
            return None
        return identity.address

    def _mark_agent_dir_processed(self, agent_dir: Path) -> None:
        conv_dir = agent_dir / "conversations"
        if not conv_dir.is_dir():
            return
        for path in conv_dir.glob("*.jsonl"):
            self._processed.add((agent_dir.name, path.name))

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
