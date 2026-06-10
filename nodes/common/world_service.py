"""Persistent World service — Phase 1 of native world-model integration.

A long-lived holder of one ``World`` instance per RPB (jurisdiction). Lives
at the node level for the lifetime of the deployment. Replaces the prior
adapter pattern that created a fresh ``World`` per task and discarded it.

Design contract
---------------

  - One ``WorldService`` per RPB address. Wired into ``AutonetService.start``.
  - Thread-safe via ``RLock``. Single writer at a time; multiple readers
    serialize through the same lock for now (will split into reader/writer
    locks at Phase 6 if inference probe latency forces it).
  - Persistence: atomic JSON snapshot to ``<root>/{rpb}/snapshot.json`` plus
    append-only event log at ``<root>/{rpb}/events.jsonl``. On restart, load
    snapshot then replay any events past the snapshot boundary.

Phase 1 is deliberately scoped to single-node, no chain, no probe protocol.
The ``submit_events`` method is the only mutating entry point and is shaped
so that Phase 2's ``submit_probe`` can layer on top without changing it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from world_model.generalized import (
    Observation,
    World,
    default_locator,
    equilibrate,
)
from world_model.generalized.scope import scope_for_observation
from world_model.generalized.equilibrate import equilibrate_continuous_exploration
from world_model.models.tree import Position

from .world_model_substrate.adapter import (
    _all_node_ids,
    _EventRecorder,
    build_charter_world,
)
from .world_model_substrate.aggregate import apply_events
from .world_model_substrate.events import snapshot_node_scores
from .world_model_substrate.mint_gate import (
    DEFAULT_CHARTER_IDS,
    apply_mint_gate,
)
from .world_model_substrate.reconcile import (
    EpochSnapshots,
    apply_emission_pool,
    reconcile_epoch,
)
from .world_persistence import (
    PersistenceConfig,
    WorldPersistence,
)


logger = logging.getLogger(__name__)


_DEFAULT_BANDWIDTH = 1.5
# Phase 2.2 (dim_sweep): 64 retains 95% of native-384 categorical
# separation, 32 drops to 84%, 16 to 73%. Previously 1024 — wasteful
# without measurable gain. Persisted 1024-dim snapshots remain
# replayable: the existing pad/truncate logic in submit_observation
# / submit_work_units zero-pads or truncates incoming coords to
# self.embedding_dim, so a service constructed at 64 reading old
# 1024-dim events will truncate them at the boundary, and a service
# constructed at 1024 reading new 64-dim events will zero-pad.
_DEFAULT_EMBEDDING_DIM = 64
_DEFAULT_EQUILIBRATE_ROUNDS = 8
_DEFAULT_EQUILIBRATE_TOLERANCE = 1e-3

# Scoped equilibrate: on the per-observation hot path, only re-act
# tendencies whose bandwidth the observation's coords fall within.
# Drops per-event cost from O(T^2 * N) to O(T_local * K). See
# POST_AUTONET_FINDINGS.md for the seed-time problem this fixes and
# verify_scoped_tier3b.py for the H4-preserved verification.
# Env override AUTONET_SCOPED_EQUILIBRATE=0 disables; default on.
_SCOPED_EQUILIBRATE_ENABLED = (
    os.environ.get("AUTONET_SCOPED_EQUILIBRATE", "1") != "0"
)

# Slow Lindblad exploration pass: surfaces cross-domain links between
# sub-claims that aren't bandwidth-neighbors. Writeback to root scores
# is suppressed inside the kernel — what it emits is cross-link stakes
# between sub-claim pairs. Those are local enrichment, NOT part of the
# canonical replay state that mint computation depends on, so it's
# safe for daemons to run it adaptively without consensus divergence.
#
# Activity-based trigger: run after every N submitted observations.
# Env: AUTONET_LINDBLAD_EXPLORE_EVERY=<int> (0 disables; default 0).
_LINDBLAD_EXPLORE_EVERY = int(
    os.environ.get("AUTONET_LINDBLAD_EXPLORE_EVERY", "0") or "0"
)


def _hash_problem_resolution(problem: str, resolution: str) -> str:
    import hashlib, json as _json
    payload = _json.dumps(
        {"p": (problem or "")[:200], "r": (resolution or "")[:200]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class WorldService:
    """Long-lived holder of one persistent World, one per RPB."""

    def __init__(
        self,
        rpb_address: str,
        *,
        data_root: Optional[Path] = None,
        bandwidth: float = _DEFAULT_BANDWIDTH,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
        persistence: Optional[WorldPersistence] = None,
        snapshot_every_n_events: int = 100,
        snapshot_every_seconds: float = 60.0,
        epoch_emission_rate: Optional[float] = None,
    ):
        self.rpb_address = rpb_address
        self.bandwidth = bandwidth
        self.embedding_dim = embedding_dim
        # Fixed-emission economics: tokens minted per SECOND of epoch
        # duration, split pro-rata by post-gate contribution. None =
        # legacy uncapped mint (every claim prints). The pool for an
        # epoch is rate x (close - open) so emission is a pure time
        # schedule regardless of how often epochs close.
        self.epoch_emission_rate = epoch_emission_rate
        self._lock = threading.RLock()
        self._world: World = build_charter_world(
            bandwidth=bandwidth, embedding_dim=embedding_dim,
        )
        self._created_at = time.time()
        self._events_applied = 0
        self._events_since_snapshot = 0
        self._last_snapshot_time = self._created_at
        # Slow Lindblad exploration trigger: counts observations since
        # the last exploration pass. Daemon-local; doesn't enter
        # canonical state.
        self._obs_since_explore = 0

        # Epoch tracking (Phase 3). When an epoch is open, every event
        # we apply gets buffered in ``self._epoch_events`` so it can be
        # passed to reconcile_epoch at close.
        self._current_epoch_id: Optional[str] = None
        self._epoch_started_at: float = 0.0
        self._epoch_snapshots: Optional[EpochSnapshots] = None
        self._epoch_events: List[Dict[str, Any]] = []
        # Wall-clock arrival time per buffered event (parallel list) —
        # the candle close partitions the buffer at a retroactively
        # drawn cutoff, so each event needs an arrival stamp.
        self._epoch_event_ts: List[float] = []
        # Events that arrived after a candle cutoff roll into the next
        # epoch's buffer (they are NOT lost — paid next window).
        self._carry_events: List[Dict[str, Any]] = []
        self._carry_event_ts: List[float] = []
        # Emission accounting clock: advances to each close's effective
        # end (the cutoff, under candle close). The next epoch's pool
        # covers [clock, next cutoff], so the unminted remainder of a
        # candle window accrues to the next epoch automatically.
        self._emission_clock: float = 0.0
        self._epoch_history: List[Dict[str, Any]] = []  # closed epochs

        # Phase 12.25: cached 2D topic projection per node id, used to
        # anchor PCA axes between recomputes so the layout stays
        # stable across epochs (PCA's eigenvectors are unique up to a
        # sign, which would otherwise flip the layout).
        self._last_topic_xy: Dict[str, Any] = {}
        # Stable cluster_id per node, recomputed alongside topic_xy.
        # Persists across calls so the frontend can rely on the same
        # cluster_id for the same node across viewport refreshes.
        self._last_clusters: Dict[str, str] = {}

        # Per-subtree PCA projections, keyed by (parent_id, epoch_count).
        # Bounded LRU; epoch_count is len(self._epoch_history) at compute
        # time, so a new close invalidates lookups by changing the key.
        self._subtree_projection_cache: "OrderedDict[Tuple[str, int], Dict[str, Any]]" = OrderedDict()
        self._subtree_projection_cache_max = 64

        # Phase 4: Subscribers receive close records (for WS push,
        # logging, downstream chain reporters in Phase 5).
        self._epoch_subscribers: List[Any] = []
        # Phase 5.2: Subscribers receive events that the daemon has
        # locally applied (for federation gossip).
        self._local_event_subscribers: List[Any] = []

        if persistence is None:
            cfg = PersistenceConfig(
                rpb_address=rpb_address,
                data_root=data_root,
                snapshot_every_n_events=snapshot_every_n_events,
                snapshot_every_seconds=snapshot_every_seconds,
                embedding_dim=embedding_dim,
                bandwidth=bandwidth,
            )
            persistence = WorldPersistence(cfg)
        self._persistence = persistence

        # Try to recover from disk; if nothing exists, the fresh charter world
        # we just built is what we use.
        restored = self._persistence.try_restore()
        if restored is not None:
            self._world = restored.world
            self._events_applied = restored.events_replayed
            logger.info(
                "WorldService restored from disk: rpb=%s, %d events replayed",
                rpb_address, restored.events_replayed,
            )
        else:
            logger.info(
                "WorldService initialized fresh: rpb=%s, charter built",
                rpb_address,
            )

    # ------------------------------------------------------------------
    # Epoch boundaries (Phase 3)
    # ------------------------------------------------------------------

    def open_epoch(self, epoch_id: Optional[str] = None) -> Dict[str, Any]:
        """Mark the start of a new reward window.

        Captures a per-node score snapshot and begins buffering events
        that arrive (via submit_events / submit_observation /
        submit_work_units) until ``close_epoch`` is called.

        Returns a small descriptor.
        """
        with self._lock:
            if self._current_epoch_id is not None:
                logger.warning(
                    "open_epoch called while %s is still open; "
                    "auto-closing the prior epoch first",
                    self._current_epoch_id,
                )
                self._close_epoch_locked(apply_gate=True)

            eid = epoch_id or f"epoch_{time.time_ns()}"
            snapshots = EpochSnapshots()
            snapshots.record_start(self._world)
            self._current_epoch_id = eid
            self._epoch_started_at = time.time()
            if self._emission_clock <= 0:
                self._emission_clock = self._epoch_started_at
            self._epoch_snapshots = snapshots
            # Events rolled forward from a candle cutoff seed the new
            # epoch's buffer (already applied to the world; counted in
            # this epoch's attribution).
            self._epoch_events = list(self._carry_events)
            self._epoch_event_ts = list(self._carry_event_ts)
            self._carry_events = []
            self._carry_event_ts = []
            logger.info(
                "WorldService epoch opened: rpb=%s, epoch=%s%s",
                self.rpb_address, eid,
                f" ({len(self._epoch_events)} events rolled forward)"
                if self._epoch_events else "",
            )
            return {
                "epoch_id": eid,
                "started_at": self._epoch_started_at,
                "n_nodes_at_start": len(snapshots.start),
            }

    def checkpoint_epoch(self) -> Optional[Dict[str, Any]]:
        """Record an intermediate score snapshot inside the current
        epoch. Survival-factor computation in reconcile_epoch uses
        these intermediate readings to detect ephemeral score moves."""
        with self._lock:
            if self._current_epoch_id is None:
                return None
            assert self._epoch_snapshots is not None
            self._epoch_snapshots.record_checkpoint(self._world)
            return {
                "epoch_id": self._current_epoch_id,
                "checkpoint_index": len(self._epoch_snapshots.checkpoints) - 1,
                "n_nodes": len(self._epoch_snapshots.checkpoints[-1]),
            }

    def close_epoch(
        self,
        *,
        apply_gate: bool = True,
        gate_strength: float = 1.0,
        agent_weights: Optional[Dict[str, float]] = None,
        cutoff_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Close the open epoch and produce a reward result.

        - Records the close snapshot.
        - Runs ``reconcile_epoch`` against buffered events.
        - Optionally applies the charter mint gate (default: yes).
        - Returns the per-agent mint, per-agent novelty, and per-node
          breakdowns.

        ``cutoff_ts`` (candle close): only events that arrived at or
        before this wall-clock time count for THIS epoch; later ones
        roll into the next epoch's buffer. The emission pool covers
        time up to the cutoff, with the remainder accruing forward.
        Note the local cutoff is attribution-level (the world state
        itself is continuous); the authoritative federated close
        applies the cutoff exactly by truncating the canonical
        sequence before replay.
        """
        with self._lock:
            return self._close_epoch_locked(
                apply_gate=apply_gate,
                gate_strength=gate_strength,
                agent_weights=agent_weights,
                cutoff_ts=cutoff_ts,
            )

    @property
    def current_epoch_id(self) -> Optional[str]:
        """The id of the open epoch, or None if none is open."""
        return self._current_epoch_id

    @property
    def epoch_history(self) -> List[Dict[str, Any]]:
        """Closed-epoch records (id, mint totals, etc)."""
        return list(self._epoch_history)

    # ------------------------------------------------------------------
    # Mutating entrypoints
    # ------------------------------------------------------------------

    def submit_events(
        self,
        events: List[Dict[str, Any]],
        *,
        equilibrate_after: bool = True,
        equilibrate_rounds: int = _DEFAULT_EQUILIBRATE_ROUNDS,
        equilibrate_tolerance: float = _DEFAULT_EQUILIBRATE_TOLERANCE,
        origin: str = "local",
    ) -> Dict[str, Any]:
        """Append events to the world and (optionally) re-equilibrate.

        Events are applied via the existing ``apply_events`` machinery so
        content-addressed dedupe still works. Returns a receipt with
        before/after root scores and the equilibration round count.

        ``origin``: ``"local"`` events get fanned out to local-event
        subscribers (gossip layer republishes them). ``"remote"`` events
        skip the fan-out to prevent loops.
        """
        if not events:
            return {
                "events_applied": 0,
                "rounds": 0,
                "root_scores_before": self._read_root_scores_locked(),
                "root_scores_after": self._read_root_scores_locked(),
            }

        with self._lock:
            scores_before = self._world.root_scores()
            apply_events(self._world, events)
            rounds = 0
            if equilibrate_after:
                rounds = equilibrate(
                    self._world,
                    max_rounds=equilibrate_rounds,
                    tolerance=equilibrate_tolerance,
                )
            scores_after = self._world.root_scores()

            # Buffer for the open epoch, if any.
            self._buffer_epoch_events_locked(events)

            # Notify local-event subscribers (federation gossip).
            # Remote-origin events skip the fan-out so peer ingest
            # doesn't re-publish what it received.
            if origin == "local":
                self._broadcast_local_events_locked(events)

            # Persist events + equilibrate marker to the append-only log,
            # so restart can reproduce the same batch boundaries.
            self._persistence.append_events(
                events,
                equilibrate_after=equilibrate_after,
                equilibrate_rounds=equilibrate_rounds,
                equilibrate_tolerance=equilibrate_tolerance,
            )
            self._events_applied += len(events)
            self._events_since_snapshot += len(events)

            self._maybe_snapshot_locked()

            return {
                "events_applied": len(events),
                "rounds": rounds,
                "root_scores_before": scores_before,
                "root_scores_after": scores_after,
            }

    def equilibrate(
        self,
        *,
        max_rounds: int = _DEFAULT_EQUILIBRATE_ROUNDS,
        tolerance: float = _DEFAULT_EQUILIBRATE_TOLERANCE,
    ) -> int:
        """Run equilibration on the current world. Returns rounds used."""
        with self._lock:
            return equilibrate(self._world, max_rounds=max_rounds, tolerance=tolerance)

    def maybe_run_exploration_pass(
        self,
        *,
        every: Optional[int] = None,
        mu: float = 2.5,
        t_total: float = 8.0,
    ) -> Dict[str, Any]:
        """Slow Lindblad exploration pass — daemon-local enrichment.

        Surfaces cross-domain links between sub-claims that aren't
        bandwidth-neighbors. Root-score writeback is suppressed inside
        the kernel; what it produces is local cross-link annotation.
        Not part of canonical replay → safe to run adaptively per
        daemon without consensus divergence.

        Caller is responsible for invoking this OUTSIDE the
        per-observation hot path (no event recorder around it). The
        per-observation counter ``_obs_since_explore`` ticks inside
        ``submit_observation``; this method consults it.

        Returns ``{"ran": bool, "obs_since_last": int, ...}``.
        """
        threshold = every if every is not None else _LINDBLAD_EXPLORE_EVERY
        with self._lock:
            obs_since = self._obs_since_explore
            if threshold <= 0 or obs_since < threshold:
                return {"ran": False, "obs_since_last": obs_since,
                        "threshold": threshold}
            equilibrate_continuous_exploration(
                self._world, mu=mu, t_total=t_total,
            )
            self._obs_since_explore = 0
            return {"ran": True, "obs_since_last": obs_since,
                    "threshold": threshold}

    def submit_observation(
        self,
        observation: Observation,
        *,
        agent_id: str = "default",
        sprout_under_charter: bool = True,
        sprout_rootless: bool = False,
        equilibrate_rounds: int = _DEFAULT_EQUILIBRATE_ROUNDS,
        equilibrate_tolerance: float = _DEFAULT_EQUILIBRATE_TOLERANCE,
    ) -> Dict[str, Any]:
        """Apply a single observation to the world.

        - ``sprout_under_charter``: cross-tendency edge discovery picks up
          the observation under whichever charter root is closest in the
          first 4 dims (engine handles this automatically when the obs
          has nontrivial charter projection).
        - ``sprout_rootless``: also create a rootless usefulness node at
          the observation's coordinates so the locator can retrieve it
          later. Use this for work-unit observations whose charter
          projection is incidental but whose embedding tail carries the
          procedural signal.

        Persists ONLY the causal events (observation_added,
        explicit-sprout). The sub-claims derived during equilibration
        are NOT persisted: replay re-runs equilibrate at the marker
        and derives them deterministically.
        """
        with self._lock:
            recorder = _EventRecorder(agent_id=agent_id)
            seq_counter = [0]

            def _next_seq() -> int:
                seq_counter[0] += 1
                return seq_counter[0]

            # Snapshot node ids before any mutation so we can capture
            # derived sprouts after equilibrate.
            before_ids = _all_node_ids(self._world)

            # 1. Add observation (causal event).
            self._world.add_observation(observation)
            recorder.observation_added(observation)

            # 2. Explicit rootless sprout, if requested. Emit a
            # SubClaimSprouted event for the explicit sprout so replay
            # creates the same node before equilibrate runs.
            from .world_model_substrate.events import SubClaimSprouted
            if sprout_rootless:
                target = self._closest_tendency_locked(observation.coords)
                if target is not None:
                    axis = self._axis_from_coords(observation.coords) \
                        or target.polarity_axis
                    new_node = target.sprout_child(
                        parent_node_id=target.tree.root_node.id,
                        position=Position.PRO,
                        anchor=tuple(observation.coords),
                        polarity_axis=tuple(axis),
                        observation=observation,
                        content=observation.label or "",
                        world=self._world,
                    )
                    recorder._seq += 1
                    recorder.events.append(SubClaimSprouted(
                        seq=recorder._seq,
                        author_agent=agent_id,
                        tendency_id=target.id,
                        parent_id=target.tree.root_node.id,
                        node_id=new_node.id,
                        position=Position.PRO.value,
                        coords=list(observation.coords),
                        polarity_axis=list(axis),
                        content=observation.label or "",
                    ))

            # 3. Equilibrate. The events derived here are NOT persisted
            # (replay's marker re-derives them), but they ARE captured
            # for the open epoch's attribution buffer — that's how mint
            # gets credited to the agent whose observation triggered
            # the cross-tendency sprouts.
            #
            # Scoped equilibrate (POST_AUTONET_FINDINGS.md): when the
            # observation's coords land inside one or more tendencies'
            # bandwidth, only those tendencies re-act this round. Out-
            # of-scope tendencies' state is preserved as-is. Drops
            # per-event cost from O(T^2*N) to O(T_local*K).
            scope = None
            if _SCOPED_EQUILIBRATE_ENABLED:
                scope = scope_for_observation(self._world, observation)
            rounds = 0
            if scope is None or scope:
                rounds = equilibrate(
                    self._world,
                    max_rounds=equilibrate_rounds,
                    tolerance=equilibrate_tolerance,
                    scope=scope,
                )
            recorder.sub_claims_after_equilibrate(self._world, before_ids)

            # 4. Split: causal_events go to the persistence log
            # (the explicit sprout is causal, its consequences are not).
            # The full event list (causal + derived) goes to the epoch
            # attribution buffer.
            full_events = [e.to_dict() for e in recorder.events]
            causal_events = [
                ev for ev in full_events
                if ev.get("kind") == "observation_added"
                or (
                    ev.get("kind") == "sub_claim_sprouted"
                    and sprout_rootless
                    # The explicit sprout came right after the obs (seq 2)
                    # — anything beyond that is derived.
                    and ev.get("seq") == 2
                )
            ]
            self._buffer_epoch_events_locked(full_events)
            # Federation: republish causal events (the canonical record)
            # to peers. submit_observation is always locally originated.
            self._broadcast_local_events_locked(causal_events)
            self._persistence.append_events(
                causal_events,
                equilibrate_after=True,
                equilibrate_rounds=equilibrate_rounds,
                equilibrate_tolerance=equilibrate_tolerance,
            )
            self._events_applied += len(causal_events)
            self._events_since_snapshot += len(causal_events)
            self._obs_since_explore += 1
            self._maybe_snapshot_locked()

            return {
                "events_applied": len(causal_events),
                "rounds": rounds,
                "root_scores": dict(self._world.root_scores()),
            }

    def submit_work_units(
        self,
        work_units: List[Any],
        *,
        agent_id: str = "default",
        embedder: Optional[Any] = None,
        equilibrate_rounds: int = _DEFAULT_EQUILIBRATE_ROUNDS,
        equilibrate_tolerance: float = _DEFAULT_EQUILIBRATE_TOLERANCE,
    ) -> Dict[str, Any]:
        """Submit a list of (problem, resolution, outcome) tuples.

        Each work unit is converted to an Observation in the
        concatenated coordinate space:
          - charter head: zeros (Phase 2 wires the LLM-binary-flag
            adapter at a later step; for now usefulness signal only)
          - embedding tail: from the usefulness embedder

        Each observation is submitted via ``submit_observation`` with
        ``sprout_rootless=True`` so the locator can find them later.
        Outcome controls position (PRO when accepted+kept >= 0).

        Returns a summary receipt with per-unit event counts and
        score deltas.
        """
        from .world_model_substrate.usefulness_coords import (
            default_usefulness_embedder,
            coords_for_problem_resolution,
        )

        if not work_units:
            return {
                "units_processed": 0,
                "events_appended": 0,
                "rounds": 0,
            }

        if embedder is None:
            embedder = default_usefulness_embedder(dim=self.embedding_dim)

        with self._lock:
            scores_before = self._world.root_scores()
            total_events = 0
            total_rounds = 0

            for unit in work_units:
                problem, resolution, outcome = unit
                tail = coords_for_problem_resolution(
                    problem, resolution, embedder=embedder,
                )
                # Pad/truncate the embedding to match self.embedding_dim.
                tail_t = tuple(tail)[: self.embedding_dim]
                if len(tail_t) < self.embedding_dim:
                    tail_t = tail_t + (0.0,) * (self.embedding_dim - len(tail_t))
                # Charter head: zeros for now. (Tier 3A LLM-binary-flag
                # integration will fill this in at a follow-up.)
                charter_head = (0.0, 0.0, 0.0, 0.0)
                coords = charter_head + tail_t

                obs_id = "wu_" + _hash_problem_resolution(problem, resolution)
                label = (problem or "")[:80]
                obs = Observation(id=obs_id, coords=coords, label=label)

                receipt = self.submit_observation(
                    obs,
                    agent_id=agent_id,
                    sprout_under_charter=True,
                    sprout_rootless=True,
                    equilibrate_rounds=equilibrate_rounds,
                    equilibrate_tolerance=equilibrate_tolerance,
                )
                total_events += receipt["events_applied"]
                total_rounds += receipt["rounds"]

            scores_after = self._world.root_scores()
            return {
                "units_processed": len(work_units),
                "events_appended": total_events,
                "rounds": total_rounds,
                "root_scores_before": scores_before,
                "root_scores_after": scores_after,
            }

    def coords_for_text(
        self,
        text: str,
        *,
        embedder: Optional[Any] = None,
        charter_head: Optional[Tuple[float, ...]] = None,
    ) -> Tuple[float, ...]:
        """Convert a free-form text query into a coord vector matching
        this WorldService's embedding_dim.

        The result is shaped ``[charter_head_N | embedding_tail_M]``
        where N == ``adapter.N_DIMS`` (the active charter dimensionality)
        and M == self.embedding_dim:

          - ``charter_head``: optional N-tuple to seed the alignment
            axes. Defaults to all zeros. Phase 6.4+ may set non-zero
            values here based on charter-axis classification of the
            query text.
          - ``embedding_tail``: from ``coords_for_query`` using the
            project's default usefulness embedder. Truncated or
            zero-padded to match self.embedding_dim.

        Deterministic given the same embedder and text. Different
        texts produce different coords assuming the embedder is
        non-degenerate.
        """
        from .world_model_substrate.adapter import N_DIMS
        from .world_model_substrate.usefulness_coords import (
            coords_for_query,
            default_usefulness_embedder,
        )
        if embedder is None:
            embedder = default_usefulness_embedder(dim=self.embedding_dim)
        head = tuple(charter_head) if charter_head is not None else (0.0,) * N_DIMS
        if len(head) != N_DIMS:
            raise ValueError(f"charter_head must be length {N_DIMS}, got {len(head)}")
        tail = tuple(coords_for_query(text, embedder=embedder))
        # Match self.embedding_dim: truncate or zero-pad.
        if len(tail) > self.embedding_dim:
            tail = tail[: self.embedding_dim]
        elif len(tail) < self.embedding_dim:
            tail = tail + (0.0,) * (self.embedding_dim - len(tail))
        return head + tail

    def probe_inference(
        self,
        coords: Any,
        *,
        mode: str = "general",
        max_results: int = 16,
        max_distance: Optional[float] = None,
        descendants_per_node: int = 3,
    ) -> Dict[str, Any]:
        """Run a transient inference probe against the persistent world.

        Phase 6.1: read-only. The general-mode probe locates the
        query's region by coordinate distance and renders it. No
        write side effects — the world isn't mutated, no observation
        is added, no equilibration runs, no events are persisted, no
        epoch buffer is touched.

        Returns a dict with:
          - mode: the mode used.
          - region: the located ``Region`` as a list of
            ``{node_id, tendency_id, distance, score, label}`` dicts.
          - render: the ``render(world, region)`` structured output
            (ancestors, descendants, scores per located node).
          - n_results: how many nodes the locator returned.

        Locator parameters (``max_results``, ``max_distance``,
        ``descendants_per_node``) are passed through; defaults are
        tuned for "give me up to 16 nearby nodes with their immediate
        ancestor chain."

        Latency: this path acquires the WorldService lock briefly for
        a coordinate read but does not mutate state. Phase 6+ may
        want a reader-writer lock if probe contention shows up.
        """
        if mode not in ("general", "alignment"):
            raise ValueError(f"unknown inference mode: {mode!r}")

        coords_t = tuple(coords)

        from world_model.generalized import (
            CoordinateLocator,
            render as _render,
        )

        with self._lock:
            if mode == "alignment":
                # Phase 6.1: not yet implemented. Alignment mode needs
                # a transient observation + brief equilibrate + rollback,
                # which is more invasive. Coming when first needed.
                raise NotImplementedError(
                    "alignment mode probe is deferred — Phase 6.1 ships "
                    "general mode only"
                )

            locator = CoordinateLocator(
                max_distance=max_distance,
                max_results=max_results,
            )
            region = locator(self._world, coords_t)
            render_out = _render(
                self._world, region,
                descendants_per_node=descendants_per_node,
            )

            # Build the per-node region summary for callers that want
            # something simpler than the full render output.
            region_summary: List[Dict[str, Any]] = []
            for tendency_id, node_id, distance in region:
                tendency = self._world.tendencies.get(tendency_id)
                if tendency is None:
                    continue
                node = tendency.tree.get_node(node_id)
                if node is None:
                    continue
                region_summary.append({
                    "node_id": node_id,
                    "tendency_id": tendency_id,
                    "distance": float(distance),
                    "score": float(getattr(node, "net_score", 0.0)),
                    "label": getattr(node, "content", "") or "",
                })

            return {
                "mode": mode,
                "n_results": len(region_summary),
                "region": region_summary,
                "render": render_out,
            }

    def locate(
        self,
        coords: Any,
        *,
        max_results: int = 16,
        max_distance: Optional[float] = None,
        include_visualization_fields: bool = False,
        recent_mint_epochs: int = 5,
    ) -> List[Dict[str, Any]]:
        """Coordinate-distance retrieval against the persistent world.

        Returns a list of dicts. With ``include_visualization_fields``
        the dict is enriched with the absolute charter coordinates,
        ``parent_id``, and ``recent_mint`` aggregated across the last
        ``recent_mint_epochs`` closed epochs — exactly what a 2D
        projection layer needs to render a node without doing a second
        round-trip per node.

        Uses the engine's ``default_locator`` (coordinate proximity
        with a keyword fallback).
        """
        from world_model.generalized import CoordinateLocator
        with self._lock:
            locator = CoordinateLocator(
                max_distance=max_distance,
                max_results=max_results,
            )
            region = locator(self._world, tuple(coords))
            recent_mint_map: Dict[str, float] = {}
            if include_visualization_fields and recent_mint_epochs > 0:
                recent = list(reversed(self._epoch_history[-recent_mint_epochs:]))
                for record in recent:
                    for nid, mint in (record.get("node_mint") or {}).items():
                        try:
                            recent_mint_map[nid] = recent_mint_map.get(nid, 0.0) + float(mint)
                        except (TypeError, ValueError):
                            continue
            out: List[Dict[str, Any]] = []
            for tendency_id, node_id, distance in region:
                tendency = self._world.tendencies.get(tendency_id)
                if tendency is None:
                    continue
                node = tendency.tree.get_node(node_id)
                if node is None:
                    continue
                entry: Dict[str, Any] = {
                    "node_id": node_id,
                    "tendency_id": tendency_id,
                    "label": getattr(node, "content", "") or "",
                    "distance": distance,
                    "score": getattr(node, "net_score", 0.0),
                }
                if include_visualization_fields:
                    node_coords = self._node_coords_locked(node, tendency)
                    if node_coords is not None:
                        entry["coords"] = node_coords
                    entry["parent_id"] = getattr(node, "parent_id", None)
                    entry["recent_mint"] = recent_mint_map.get(node_id, 0.0)
                    # Per-tendency scores: each charter root contributes
                    # its own intrinsic score, computed via the tendency-
                    # aware walk so cross-tendency edges don't pollute it.
                    entry["scores"] = self._per_tendency_scores_locked(node)
                    cid = self._last_clusters.get(node_id)
                    if cid is not None:
                        entry["cluster_id"] = cid
                out.append(entry)
            return out

    def _node_coords_locked(self, node: Any, tendency: Any) -> Optional[list]:
        """Resolve a node's full charter+embedding coordinates.

        Caller must hold ``self._lock``. The node's coords live on its
        underlying ``Observation``, looked up via ``observation_id`` —
        ``node.content`` is just the label string. Falls back to the
        tendency anchor (rare; only if the node sprouted without a
        backing observation).
        """
        obs_id = getattr(node, "observation_id", None)
        if obs_id:
            obs = self._world.observations.get(obs_id)
            if obs is not None:
                coords = getattr(obs, "coords", None)
                if coords is not None:
                    return list(coords)
        anchor = getattr(tendency, "anchor", None)
        return list(anchor) if anchor is not None else None

    @staticmethod
    def _iter_tree_nodes(tree: Any) -> List[Any]:
        """Best-effort node enumeration across Tree implementations.

        Tries ``all_nodes`` first (the canonical Tree class), then a
        few fallback attribute names used by alternate implementations
        and serialized stand-ins.
        """
        nodes_iter = (
            getattr(tree, "all_nodes", None)
            or getattr(tree, "iter_nodes", None)
            or getattr(tree, "nodes", None)
            or getattr(tree, "_nodes", None)
        )
        if callable(nodes_iter):
            try:
                return list(nodes_iter())
            except Exception:
                return []
        if nodes_iter is None:
            return []
        try:
            if isinstance(nodes_iter, dict):
                return list(nodes_iter.values())
            return list(nodes_iter)
        except Exception:
            return []

    def _descendants_of_locked(self, parent_id: str) -> set[str]:
        """Walk every tendency tree to collect transitive descendants
        of ``parent_id``. Returns a set of node ids.

        Used by the viewport-scoped query when the user has drilled
        into a subtree. Caller must hold ``self._lock``.
        """
        descendants: set[str] = set()
        parents: Dict[str, str] = {}
        for tendency in self._world.tendencies.values():
            tree = getattr(tendency, "tree", None)
            if tree is None:
                continue
            for n in self._iter_tree_nodes(tree):
                if n is None:
                    continue
                nid = getattr(n, "id", None)
                pid = getattr(n, "parent_id", None)
                if nid is not None and pid is not None:
                    parents[nid] = pid
        # BFS from parent_id through the parent_id -> children inverse.
        children: Dict[str, list[str]] = {}
        for nid, pid in parents.items():
            children.setdefault(pid, []).append(nid)
        frontier = list(children.get(parent_id, []))
        while frontier:
            nid = frontier.pop()
            if nid in descendants:
                continue
            descendants.add(nid)
            frontier.extend(children.get(nid, []))
        return descendants

    def _per_tendency_scores_locked(self, node: Any) -> Dict[str, float]:
        """Compute the node's score under each charter tendency.

        Caller must hold ``self._lock``. Returns a dict keyed by
        tendency_id with the tendency-tree-aware intrinsic score for
        the node. Substrate-honest replacement for the single
        ``net_score`` scalar — a node lives in 6D charter space and
        has six concurrent verdicts.
        """
        from world_model.generalized.tendency import _intrinsic_score_in_tendency
        scores: Dict[str, float] = {}
        for tendency_id in self._world.tendencies:
            try:
                scores[tendency_id] = float(
                    _intrinsic_score_in_tendency(node, tendency_id)
                )
            except Exception:
                scores[tendency_id] = 0.0
        return scores

    def list_nodes_for_visualization(
        self,
        *,
        max_nodes: int = 200,
        recent_mint_epochs: int = 5,
        bin_size: Optional[float] = None,
        region_xy: Optional[Tuple[float, float]] = None,
        region_radius: Optional[float] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return all nodes (or spatial bins) suitable for rendering.

        Unlike ``locate`` (which retrieves around a single coord point),
        this returns a global view: the top-N nodes by recent mint plus
        their charter coordinates and attribution-ready fields.

        ``bin_size`` enables density-first LOD: when set, nodes are
        grouped into spatial bins of side ``bin_size`` in charter
        space; one aggregate marker is returned per non-empty bin
        instead of individual nodes. Bin coords are the centroid of
        their members; ``count`` and ``total_recent_mint`` are the
        bin's aggregate. Used at low zoom levels to avoid sending
        thousands of dots.

        Lazy-loading hooks (Phase 12.26b):
          * ``parent_id`` — restrict to direct + transitive descendants
            of that node. Used when the user drills into a subtree.
          * ``region_xy`` + ``region_radius`` — restrict to nodes whose
            ``topic_xy`` falls within ``region_radius`` of ``region_xy``
            in normalized topic-space. Used as the visualization pans
            and zooms.

          When both are set, ``parent_id`` is applied first (it's a
          hard structural filter); the region disk applies on the
          remaining set. When neither is set, the full top-N global
          view is returned (legacy behavior).

        Returns:
          {
            "mode": "nodes" | "bins",
            "items": [
              {coords, recent_mint, score, parent_id, cluster_id, ...}
              or bin aggregates
            ],
            "epochs_considered": int,
          }
        """
        with self._lock:
            recent = list(reversed(self._epoch_history[-recent_mint_epochs:]))
            recent_mint_map: Dict[str, float] = {}
            for record in recent:
                for nid, mint in (record.get("node_mint") or {}).items():
                    try:
                        recent_mint_map[nid] = recent_mint_map.get(nid, 0.0) + float(mint)
                    except (TypeError, ValueError):
                        continue

            # Phase 12.26b parent_id filter — pre-compute the descendant
            # set of the requested anchor so we can skip non-descendants
            # during the tree walk. Walks the parent_id chain from each
            # candidate up to either the requested parent or a root.
            allowed_descendants: Optional[set[str]] = None
            if parent_id:
                allowed_descendants = self._descendants_of_locked(parent_id)

            # Walk every tendency's tree, collect node entries.
            entries: List[Dict[str, Any]] = []
            for tendency_id, tendency in self._world.tendencies.items():
                tree = getattr(tendency, "tree", None)
                if tree is None:
                    continue
                for node in self._iter_tree_nodes(tree):
                    if node is None:
                        continue
                    node_id = getattr(node, "id", None)
                    if node_id is None:
                        continue
                    # Skip the tree root claim — it's the tendency
                    # anchor itself, not a substrate node users care to
                    # see in the visualization.
                    if getattr(node, "is_root", False):
                        continue
                    if allowed_descendants is not None \
                            and node_id not in allowed_descendants:
                        continue
                    coords = self._node_coords_locked(node, tendency)
                    if coords is None:
                        continue
                    label_text = getattr(node, "content", None)
                    if not isinstance(label_text, str):
                        label_text = getattr(label_text, "text", "") or ""
                    entries.append({
                        "node_id": node_id,
                        "tendency_id": tendency_id,
                        "label": label_text,
                        "coords": coords,
                        "score": getattr(node, "net_score", 0.0),
                        "scores": self._per_tendency_scores_locked(node),
                        "parent_id": getattr(node, "parent_id", None),
                        "recent_mint": recent_mint_map.get(node_id, 0.0),
                    })

        # Sort by recent mint desc, take top N for visualization.
        entries.sort(key=lambda e: e["recent_mint"], reverse=True)
        entries = entries[:max_nodes]

        # Phase 12.25: compute topic-space 2D projection (PCA on the
        # embedding tail) so the topic-view sister visualization can
        # render nodes by content similarity. Anchor with the
        # previous projection so the layout doesn't flip across runs.
        if entries:
            from .topic_projection import (
                assign_topic_clusters,
                project_topic_2d,
            )
            topic_xy = project_topic_2d(
                ((e["node_id"], e["coords"]) for e in entries),
                charter_dim=6,
                previous_xy=self._last_topic_xy,
            )
            # Stable proximity-merge clustering on the projected
            # coords. Reuses prior cluster ids when their members fall
            # together again, so the frontend cluster identity persists
            # across viewport refreshes.
            clusters = assign_topic_clusters(
                topic_xy,
                previous_clusters=self._last_clusters,
            )
            for entry in entries:
                xy = topic_xy.get(entry["node_id"])
                if xy is not None:
                    entry["topic_xy"] = [xy[0], xy[1]]
                cid = clusters.get(entry["node_id"])
                if cid is not None:
                    entry["cluster_id"] = cid
            # Update caches for the next call.
            self._last_topic_xy = dict(topic_xy)
            self._last_clusters = dict(clusters)

        # Phase 12.26b region filter — keep only nodes whose topic_xy
        # falls within the requested disk. Applied AFTER projection
        # so the cluster ids and topic positions remain consistent
        # with what neighboring viewports see (we'd otherwise project
        # different subsets onto different axes per viewport call).
        if region_xy is not None and region_radius is not None:
            cx, cy = float(region_xy[0]), float(region_xy[1])
            r2 = float(region_radius) ** 2
            kept: List[Dict[str, Any]] = []
            for entry in entries:
                xy = entry.get("topic_xy")
                if xy is None:
                    continue
                dx = xy[0] - cx
                dy = xy[1] - cy
                if dx * dx + dy * dy <= r2:
                    kept.append(entry)
            entries = kept

        if bin_size is None or bin_size <= 0:
            return {
                "mode": "nodes",
                "items": entries,
                "epochs_considered": len(recent),
            }

        # Density-first LOD: bin in charter space.
        bins: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for entry in entries:
            coords = entry["coords"]
            key = tuple(int(c // bin_size) for c in coords)
            bin_entry = bins.get(key)
            if bin_entry is None:
                bin_entry = {
                    "coords": list(coords),  # running mean
                    "count": 1,
                    "total_recent_mint": entry["recent_mint"],
                    "score_sum": entry["score"],
                }
                bins[key] = bin_entry
            else:
                n = bin_entry["count"]
                bin_entry["coords"] = [
                    (bin_entry["coords"][i] * n + coords[i]) / (n + 1)
                    for i in range(len(coords))
                ]
                bin_entry["count"] = n + 1
                bin_entry["total_recent_mint"] += entry["recent_mint"]
                bin_entry["score_sum"] += entry["score"]
        for bin_entry in bins.values():
            bin_entry["mean_score"] = bin_entry["score_sum"] / bin_entry["count"]
            del bin_entry["score_sum"]
        return {
            "mode": "bins",
            "items": list(bins.values()),
            "epochs_considered": len(recent),
        }

    def compute_subtree_projection(
        self,
        parent_id: str,
        *,
        max_nodes: int = 200,
        recent_mint_epochs: int = 5,
    ) -> Dict[str, Any]:
        """Per-subtree PCA projection for zoom-driven exploration.

        Returns the same shape as ``list_nodes_for_visualization``'s
        ``mode="nodes"`` branch, but restricted to ``parent_id``'s
        transitive descendants and projected onto a fresh PCA fit on
        only that subset's embedding tails. The subtree's axes are
        independent of the network-level projection — there's no
        ``previous_xy`` anchor because the two coordinate systems are
        not comparable.
        """
        with self._lock:
            cache_key = (parent_id, len(self._epoch_history))
            cached = self._subtree_projection_cache.get(cache_key)
            if cached is not None:
                self._subtree_projection_cache.move_to_end(cache_key)
                return cached

            recent = list(reversed(self._epoch_history[-recent_mint_epochs:]))
            recent_mint_map: Dict[str, float] = {}
            for record in recent:
                for nid, mint in (record.get("node_mint") or {}).items():
                    try:
                        recent_mint_map[nid] = recent_mint_map.get(nid, 0.0) + float(mint)
                    except (TypeError, ValueError):
                        continue

            descendants = self._descendants_of_locked(parent_id)
            if not descendants:
                empty = {
                    "mode": "nodes",
                    "items": [],
                    "epochs_considered": len(recent),
                    "parent_id": parent_id,
                }
                self._cache_subtree_projection_locked(cache_key, empty)
                return empty

            entries: List[Dict[str, Any]] = []
            for tendency_id, tendency in self._world.tendencies.items():
                tree = getattr(tendency, "tree", None)
                if tree is None:
                    continue
                for node in self._iter_tree_nodes(tree):
                    if node is None:
                        continue
                    node_id = getattr(node, "id", None)
                    if node_id is None or node_id not in descendants:
                        continue
                    if getattr(node, "is_root", False):
                        continue
                    coords = self._node_coords_locked(node, tendency)
                    if coords is None:
                        continue
                    label_text = getattr(node, "content", None)
                    if not isinstance(label_text, str):
                        label_text = getattr(label_text, "text", "") or ""
                    entries.append({
                        "node_id": node_id,
                        "tendency_id": tendency_id,
                        "label": label_text,
                        "coords": coords,
                        "score": getattr(node, "net_score", 0.0),
                        "scores": self._per_tendency_scores_locked(node),
                        "parent_id": getattr(node, "parent_id", None),
                        "recent_mint": recent_mint_map.get(node_id, 0.0),
                    })

            entries.sort(key=lambda e: e["recent_mint"], reverse=True)
            entries = entries[:max_nodes]

            if entries:
                from .topic_projection import (
                    assign_topic_clusters,
                    project_topic_2d,
                )
                topic_xy = project_topic_2d(
                    ((e["node_id"], e["coords"]) for e in entries),
                    charter_dim=6,
                    previous_xy=None,
                )
                clusters = assign_topic_clusters(
                    topic_xy,
                    previous_clusters=None,
                )
                for entry in entries:
                    xy = topic_xy.get(entry["node_id"])
                    if xy is not None:
                        entry["topic_xy"] = [xy[0], xy[1]]
                    cid = clusters.get(entry["node_id"])
                    if cid is not None:
                        entry["cluster_id"] = cid

            result = {
                "mode": "nodes",
                "items": entries,
                "epochs_considered": len(recent),
                "parent_id": parent_id,
            }
            self._cache_subtree_projection_locked(cache_key, result)
            return result

    def _cache_subtree_projection_locked(
        self,
        key: Tuple[str, int],
        value: Dict[str, Any],
    ) -> None:
        cache = self._subtree_projection_cache
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._subtree_projection_cache_max:
            cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    def read_root_scores(self) -> Dict[str, float]:
        with self._lock:
            return self._read_root_scores_locked()

    def read_node_scores(self) -> Dict[str, float]:
        """All node-level scores (not just charter roots)."""
        with self._lock:
            return snapshot_node_scores(self._world)

    # ------------------------------------------------------------------
    # Epoch projection surface (Phase 4)
    # ------------------------------------------------------------------

    def subscribe_local_events(self, handler: Any) -> None:
        """Register a handler invoked with each batch of events that this
        daemon applies locally (via submit_events / submit_observation /
        submit_work_units).

        Used by the federation gossip layer (Phase 5.2) to publish
        local activity to peer daemons. Handlers receive the list of
        event dicts as they were applied; the gossip layer signs and
        publishes them as an EventBatch.

        Phase 5.2 only fires this for **locally originated** batches;
        events arriving FROM peers (via remote submit_events) skip the
        broadcast to avoid loops. The world_service tracks origin via
        the ``_locally_originated`` flag on submit_events.
        """
        with self._lock:
            self._local_event_subscribers.append(handler)

    def unsubscribe_local_events(self, handler: Any) -> None:
        with self._lock:
            try:
                self._local_event_subscribers.remove(handler)
            except ValueError:
                pass

    def subscribe_epoch_closed(self, handler: Any) -> None:
        """Register a handler invoked whenever an epoch closes.

        Handler signature: ``handler(record: dict) -> None``. Exceptions
        in handlers are logged and isolated; one bad handler doesn't
        break others.
        """
        with self._lock:
            self._epoch_subscribers.append(handler)

    def unsubscribe_epoch_closed(self, handler: Any) -> None:
        with self._lock:
            try:
                self._epoch_subscribers.remove(handler)
            except ValueError:
                pass

    def read_epoch(self, epoch_id: str) -> Optional[Dict[str, Any]]:
        """Read a closed-epoch record by id. Returns None if missing.

        Returned record has ``authoritative=False`` until Phase 5
        federation stamps it.
        """
        with self._lock:
            # Prefer in-memory history (fastest path).
            for rec in reversed(self._epoch_history):
                if rec.get("epoch_id") == epoch_id:
                    return dict(rec)
            # Fall back to disk for older epochs.
            return self._persistence.read_epoch_record(epoch_id)

    def read_recent_epochs(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return up to ``n`` most recent closed epochs (latest first)."""
        with self._lock:
            recent = list(reversed(self._epoch_history[-n:]))
        if len(recent) >= n:
            return recent
        # Fall back to disk for older records.
        on_disk: List[Dict[str, Any]] = []
        for path in reversed(self._persistence.list_epoch_records()):
            if any(rec.get("epoch_id") == path.stem for rec in recent):
                continue  # already in memory
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    on_disk.append(json.load(fh))
            except Exception:
                continue
            if len(recent) + len(on_disk) >= n:
                break
        return recent + on_disk

    def read_agent_projection(self, agent_id: str, *, last_n_epochs: int = 10) -> Dict[str, Any]:
        """Aggregate this agent's projection (mint + novelty) across the
        last N closed epochs.

        Projection-level: not yet authoritative. Once Phase 5 federation
        runs, agents read authoritative attributions from chain instead
        of this surface.
        """
        epochs = self.read_recent_epochs(n=last_n_epochs)
        total_mint = 0.0
        total_novelty = 0.0
        per_epoch: List[Dict[str, Any]] = []
        for rec in epochs:
            mint = float(rec.get("agent_mint", {}).get(agent_id, 0.0))
            novelty = float(rec.get("agent_novelty", {}).get(agent_id, 0.0))
            total_mint += mint
            total_novelty += novelty
            per_epoch.append({
                "epoch_id": rec.get("epoch_id"),
                "closed_at": rec.get("closed_at"),
                "mint": mint,
                "novelty": novelty,
                "authoritative": rec.get("authoritative", False),
            })
        return {
            "agent_id": agent_id,
            "total_mint_projection": total_mint,
            "total_novelty_projection": total_novelty,
            "epochs_considered": len(per_epoch),
            "per_epoch": per_epoch,
        }

    def read_substrate_distribution(
        self, *, last_n_epochs: int = 10,
    ) -> Dict[str, Any]:
        """Aggregate the substrate's current root scores plus per-root
        mint over the last ``n`` closed epochs.

        Used by the top-level Network visualization to show *which
        root the network's attention is on right now* and *which
        roots earned mint recently*.

        Returns:
          {
            "root_scores": {root_id: float, ...},        # current
            "root_mint_recent": {root_id: float, ...},   # cumulative over n epochs
            "epochs_considered": int,
            "total_mint_recent": float,
          }

        Per-root mint is computed by attributing each minting node to
        its closest tendency (highest stake from that tendency on the
        node). This is a best-effort projection — accurate when nodes
        are clearly under one root, approximate when nodes straddle.
        """
        with self._lock:
            scores = self._read_root_scores_locked()
            recent = list(reversed(self._epoch_history[-last_n_epochs:]))
            # Build a node-id → root-id map by walking each tendency's
            # observations / sub-claims. The world's locator does this
            # at greater fidelity, but for a UI projection we just
            # assign each node to whichever tendency has the most
            # stake on it.
            node_to_root: Dict[str, str] = {}
            try:
                for tendency_id, tendency in self._world.tendencies.items():
                    # tendencies hold the node_ids that landed under them
                    for node_id in getattr(tendency, "node_ids", []) or []:
                        # First-write-wins: a node attributed to one root
                        # stays there. Order is consistent because dict
                        # iteration is insertion-order in py3.7+.
                        node_to_root.setdefault(node_id, tendency_id)
            except Exception as e:
                logger.debug("substrate distribution: tendency walk failed: %s", e)

        # Aggregate node_mint across recent epochs and project onto roots.
        root_mint: Dict[str, float] = {rid: 0.0 for rid in scores}
        total = 0.0
        for record in recent:
            node_mint_map = record.get("node_mint") or {}
            for node_id, mint in node_mint_map.items():
                try:
                    m = float(mint)
                except (TypeError, ValueError):
                    continue
                root_id = node_to_root.get(node_id)
                if root_id and root_id in root_mint:
                    root_mint[root_id] += m
                total += m

        return {
            "root_scores": scores,
            "root_mint_recent": root_mint,
            "epochs_considered": len(recent),
            "total_mint_recent": total,
        }

    def _broadcast_epoch_closed(self, record: Dict[str, Any]) -> None:
        """Invoke all registered subscribers with the close record.
        Must be called under self._lock; handlers run synchronously."""
        for handler in list(self._epoch_subscribers):
            try:
                handler(record)
            except Exception as e:
                logger.error(
                    "epoch close subscriber failed: %s", e, exc_info=True,
                )

    def _broadcast_local_events_locked(self, events: List[Dict[str, Any]]) -> None:
        """Notify local-event subscribers (federation gossip) of a
        locally-applied batch. Must be called under self._lock."""
        if not events:
            return
        for handler in list(self._local_event_subscribers):
            try:
                handler(events)
            except Exception as e:
                logger.error(
                    "local event subscriber failed: %s", e, exc_info=True,
                )

    def stats(self) -> Dict[str, Any]:
        """Cheap snapshot of service state for monitoring."""
        with self._lock:
            n_nodes = sum(
                len(t.tree.all_nodes())
                for t in self._world.tendencies.values()
            )
            return {
                "rpb_address": self.rpb_address,
                "events_applied": self._events_applied,
                "events_since_snapshot": self._events_since_snapshot,
                "n_tendencies": len(self._world.tendencies),
                "n_nodes": n_nodes,
                "uptime_seconds": time.time() - self._created_at,
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def snapshot_now(self) -> Optional[Path]:
        """Force an immediate snapshot. Returns the snapshot path."""
        with self._lock:
            return self._snapshot_locked()

    def shutdown(self) -> None:
        """Graceful shutdown: snapshot world, close persistence handles."""
        with self._lock:
            try:
                self._snapshot_locked()
            except Exception as e:
                logger.error("snapshot during shutdown failed: %s", e)
            self._persistence.close()
        logger.info("WorldService shut down: rpb=%s", self.rpb_address)

    # ------------------------------------------------------------------
    # Internal — must be called under self._lock
    # ------------------------------------------------------------------

    def _read_root_scores_locked(self) -> Dict[str, float]:
        return dict(self._world.root_scores())

    def _close_epoch_locked(
        self,
        *,
        apply_gate: bool = True,
        gate_strength: float = 1.0,
        agent_weights: Optional[Dict[str, float]] = None,
        cutoff_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Internal: must be called under self._lock."""
        if self._current_epoch_id is None:
            logger.warning("close_epoch called with no open epoch; ignoring")
            return {
                "epoch_id": None,
                "agent_mint": {},
                "agent_novelty": {},
                "node_mint": {},
                "node_novelty": {},
                "total_mint": 0.0,
                "total_novelty": 0.0,
                "n_events": 0,
            }
        assert self._epoch_snapshots is not None

        snapshots = self._epoch_snapshots
        snapshots.record_close(self._world)

        # Candle cutoff: partition the buffer at the retroactively
        # drawn cut. Post-cut events roll forward to the next epoch.
        events_rolled = 0
        if cutoff_ts is not None and self._epoch_events:
            kept: List[Dict[str, Any]] = []
            kept_ts: List[float] = []
            for ev, ts in zip(self._epoch_events, self._epoch_event_ts):
                if ts <= cutoff_ts:
                    kept.append(ev)
                    kept_ts.append(ts)
                else:
                    self._carry_events.append(ev)
                    self._carry_event_ts.append(ts)
            events_rolled = len(self._epoch_events) - len(kept)
            events = kept
        else:
            events = list(self._epoch_events)

        result = reconcile_epoch(
            self._world,
            snapshots,
            events,
            agent_weights=agent_weights,
        )
        if apply_gate:
            result = apply_mint_gate(
                result, self._world,
                charter_ids=DEFAULT_CHARTER_IDS,
                gate_strength=gate_strength,
            )
        # Fixed-emission normalization (post-gate, so suppressed junk
        # redistributes to survivors). The pool covers the time from
        # the emission clock (last close's effective end) to this
        # close's effective end — the cutoff under candle close, so a
        # candle window's unminted remainder accrues to the next epoch.
        # Local closes are projection-level (authoritative=False), so
        # local clocks are fine HERE; the federated close must derive
        # its pool from canonical/anchored timestamps instead.
        pool_end = cutoff_ts if cutoff_ts is not None else time.time()
        if self.epoch_emission_rate is not None and self._emission_clock > 0:
            duration = max(0.0, pool_end - self._emission_clock)
            result = apply_emission_pool(
                result, self.epoch_emission_rate * duration,
            )
        self._emission_clock = max(self._emission_clock, pool_end)

        eid = self._current_epoch_id
        record = {
            "epoch_id": eid,
            "rpb_address": self.rpb_address,
            "started_at": self._epoch_started_at,
            "closed_at": time.time(),
            "n_events": len(events),
            "agent_mint": result.get("agent_mint", {}),
            "agent_novelty": result.get("agent_novelty", {}),
            "node_mint": result.get("node_mint", {}),
            "node_novelty": result.get("node_novelty", {}),
            "total_mint": result.get("total_mint", 0.0),
            "total_novelty": result.get("total_novelty", 0.0),
            "gate_applied": apply_gate,
            "cutoff_ts": cutoff_ts,
            "events_rolled_forward": events_rolled,
            "emission_pool": result.get("emission_pool"),
            # Phase 4 surfaces projection-level results only. Phase 5's
            # federated close will stamp authoritative=True on the
            # canonical version.
            "authoritative": False,
            "scope": "local",
        }
        self._epoch_history.append(record)
        try:
            self._persistence.write_epoch_record(record)
        except Exception as e:
            logger.warning("failed to persist epoch record: %s", e)
        # Notify subscribers (e.g. the daemon's WS bridge).
        self._broadcast_epoch_closed(record)

        # Reset state (carry buffers survive — open_epoch consumes them)
        self._current_epoch_id = None
        self._epoch_started_at = 0.0
        self._epoch_snapshots = None
        self._epoch_events = []
        self._epoch_event_ts = []

        # Full result is returned for the caller; record kept in history
        # is the slim version.
        result.update({
            "epoch_id": eid,
            "n_events": len(events),
        })
        logger.info(
            "WorldService epoch closed: rpb=%s, epoch=%s, "
            "%d events, total mint=%.4f, total novelty=%.4f",
            self.rpb_address, eid, len(events),
            result["total_mint"], result["total_novelty"],
        )
        return result

    def _buffer_epoch_events_locked(self, event_dicts: List[Dict[str, Any]]) -> None:
        """If an epoch is open, accumulate the events for reconciliation."""
        if self._current_epoch_id is not None and event_dicts:
            self._epoch_events.extend(event_dicts)
            now = time.time()
            self._epoch_event_ts.extend([now] * len(event_dicts))

    def _closest_tendency_locked(self, coords):
        """Tendency whose anchor has the highest cosine similarity to
        the given coords (over the dims they share). Used by
        submit_observation to pick where to anchor a sprout."""
        best = None
        best_score = -2.0
        coords_t = tuple(coords)
        for tendency in self._world.tendencies.values():
            anchor = tuple(tendency.anchor)
            n = min(len(coords_t), len(anchor))
            if n == 0:
                continue
            dot = sum(coords_t[i] * anchor[i] for i in range(n))
            mag_a = (sum(c * c for c in coords_t[:n])) ** 0.5
            mag_b = (sum(a * a for a in anchor[:n])) ** 0.5
            if mag_a == 0 or mag_b == 0:
                cos = 0.0
            else:
                cos = dot / (mag_a * mag_b)
            if cos > best_score:
                best_score = cos
                best = tendency
        return best

    @staticmethod
    def _axis_from_coords(coords):
        """L2-normalized direction from the origin to the coords. Used
        as the polarity axis for new sprouts so neighbours along the
        same direction align with them."""
        coords_t = tuple(coords)
        mag = (sum(c * c for c in coords_t)) ** 0.5
        if mag == 0:
            return None
        return tuple(c / mag for c in coords_t)

    def _maybe_snapshot_locked(self) -> None:
        cfg = self._persistence.config
        now = time.time()
        time_due = (now - self._last_snapshot_time) >= cfg.snapshot_every_seconds
        count_due = self._events_since_snapshot >= cfg.snapshot_every_n_events
        if time_due or count_due:
            self._snapshot_locked()

    def _snapshot_locked(self) -> Path:
        path = self._persistence.write_snapshot(
            self._world,
            events_applied=self._events_applied,
        )
        self._events_since_snapshot = 0
        self._last_snapshot_time = time.time()
        return path
