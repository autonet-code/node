"""
Reseed-and-equilibrate: the universal operation.

Reseeding is the only way the engine generates non-present configurations.
Past, future, alternate-present, cosmological-time-slice -- all produced
by the same call: substitute the tendencies that differ for the target
configuration, hold the rest fixed, run calibration until allocations
settle, return the new state.

The operation is a pure function over present state. Inputs are deep-
copied at the boundary; outputs are a fresh state object. No side
effects on the inputs.

Substitution interface
======================

A substitution is a total replacement: ``substitute(id, new_tendency)``.
The engine does not reconcile fields. The caller builds the replacement
to look however the target configuration requires -- mostly identical
to the original, completely different, or anywhere in between. The
engine's job is composition, not construction.

Substitutions can also add (id not yet present) or remove (replacement
is None / sentinel) tendencies. After substitution, allocations
renormalize and calibration re-equilibrates the modified set.

Convergence
===========

The calibration loop applies the existing contraction-map reallocation
iteratively until the largest absolute allocation delta falls below
``tolerance``, capped at ``max_iterations``. The contraction property
guarantees convergence in principle; the cap is a sanity belt.

Each completed reseed emits one ALLOCATION_SHIFTED event per affected
tendency, reporting the *net* shift over the whole reseed (not per
inner iteration). This keeps event counts bounded and avoids
spuriously inflating engine-time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from ..models.tendency import Tendency, TendencySet
from ..models.lineage import (
    EngineClock,
    EventType,
    Lineage,
    LineageRecorder,
    StakeWeightGraph,
)


# ---------------------------------------------------------------------------
# Present-state aggregate
# ---------------------------------------------------------------------------


@dataclass
class PresentState:
    """The full present state: tendencies + lineage + stake-weight graph.

    This is the unit reseed-and-equilibrate operates on. It is
    intentionally a value-type aggregate -- nothing here references
    the live arena, so deep-copy is straightforward.

    Reseed never mutates an input PresentState; it returns a new one.
    """

    tendencies: TendencySet
    lineages: dict[str, Lineage] = field(default_factory=dict)
    graph: StakeWeightGraph = field(default_factory=StakeWeightGraph)

    def copy(self) -> "PresentState":
        """Deep-copy the entire state. Safe to mutate the result."""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Substitutions and result
# ---------------------------------------------------------------------------


@dataclass
class Substitution:
    """One total replacement.

    - ``new_tendency`` populated -> add or replace.
    - ``new_tendency`` is None    -> remove the tendency at ``id``.

    Optionally, the caller can supply replacement edges for the
    stake-weight graph. If ``edges`` is provided, the new tendency's
    edges are set to exactly those weights (any prior edges from this
    tendency are dropped first). If ``edges`` is None, the existing
    edges are preserved -- the caller is reusing the original
    tendency's neighborhood.
    """

    id: str
    new_tendency: Optional[Tendency] = None
    edges: Optional[dict[str, float]] = None    # other_id -> weight

    @property
    def is_removal(self) -> bool:
        return self.new_tendency is None


@dataclass
class ReseedResult:
    """What a reseed produced."""

    state: PresentState
    iterations: int
    converged: bool                  # True if delta fell below tolerance before cap
    final_max_delta: float           # the largest |delta| at termination
    affected_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def reseed_and_equilibrate(
    state: PresentState,
    substitutions: list[Substitution],
    *,
    tolerance: float = 1e-4,
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    propagate_via_graph: bool = False,
    sharpen_among_ids: Optional[set[str]] = None,
    sharpen_strength: float = 1.0,
    recorder: Optional[LineageRecorder] = None,
) -> ReseedResult:
    """Apply substitutions, calibrate to a fixpoint, return the new state.

    Parameters
    ----------
    state:
        Input state. Deep-copied; not mutated.
    substitutions:
        List of total replacements (or removals). Order matters only when
        substitutions interact -- e.g., adding A and removing A in the
        same call applies them in order.
    tolerance:
        Calibration loop terminates when the largest |allocation delta|
        across one iteration falls below this.
    max_iterations:
        Hard cap. If hit, ``ReseedResult.converged`` is False but the
        operation still returns the partially-settled state.
    learning_rate:
        Per-iteration blend factor toward the substitution-implied target
        distribution. The contraction property requires 0 < lr < 1.
    propagate_via_graph:
        If True, calibration computes per-iteration targets for non-
        substituted tendencies as a weighted average of their stake-graph
        neighbors' allocations. Substituted tendencies keep the explicit
        targets the caller supplied. This is how the engine performs
        actual graph-mediated inference -- evidence injected via
        substitution propagates through the topology to reshape the
        equilibrium of the rest of the system.

        If False (default), every tendency's target is fixed at the
        substitution-implied distribution; non-substituted tendencies
        targets equal their pre-call allocations. This is the simpler
        "just settle the math" mode.
    sharpen_among_ids:
        Optional set of tendency ids that should *compete* with each
        other after each calibration step. Within this set, allocations
        are amplified by a softmax-like sharpening: the strongest
        candidate is boosted, weaker candidates are suppressed. This
        adds discriminative dynamics (winner amplification) to the
        otherwise smoothing graph-walk relaxation. Use it when you
        want classification-like readouts where a single candidate
        should dominate. None (default) means no sharpening.
    sharpen_strength:
        Temperature inverse for the softmax sharpen. 0 = no sharpening
        (uniform), 1 = standard softmax, larger = more aggressive
        winner amplification. Reasonable values are 1-10.
    recorder:
        If provided, emits one ALLOCATION_SHIFTED event per affected
        tendency at the end of calibration, reporting the net shift.
        If None, no events are emitted (still pure-function from the
        engine's perspective; caller chose to discard event provenance).

    Returns
    -------
    ReseedResult with the new state, iteration count, convergence
    status, the final max delta, and the list of tendency ids whose
    allocations changed by more than tolerance from their input values.
    """
    if not (0.0 < learning_rate < 1.0):
        raise ValueError(
            f"learning_rate must be in (0, 1) for the contraction map to "
            f"converge; got {learning_rate}"
        )
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be > 0; got {tolerance}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0; got {max_iterations}")

    new_state = state.copy()

    # Snapshot of pre-reseed allocations so we can report net deltas at the end.
    pre_alloc: dict[str, float] = {
        t.id: t.allocation for t in new_state.tendencies.all()
    }

    # 1. Apply substitutions.
    target_allocs = _apply_substitutions(new_state, substitutions)

    # 2. Renormalize post-substitution.
    new_state.tendencies.normalize()

    # 3. Calibrate. The set of "anchored" ids is the set of substituted
    #    tendencies (excluding removals); their targets are explicit. In
    #    graph-propagation mode, non-anchored tendencies' targets are
    #    computed each iteration from their stake-graph neighborhood.
    anchored_ids = {sub.id for sub in substitutions if not sub.is_removal}
    iterations, converged, final_delta = _calibrate(
        new_state.tendencies,
        target_allocs,
        tolerance=tolerance,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        propagate_via_graph=propagate_via_graph,
        graph=new_state.graph if propagate_via_graph else None,
        anchored_ids=anchored_ids if propagate_via_graph else None,
        sharpen_among_ids=sharpen_among_ids,
        sharpen_strength=sharpen_strength,
    )

    # 4. Compute affected ids (those whose allocation moved by > tolerance).
    affected: list[str] = []
    for t in new_state.tendencies.all():
        before = pre_alloc.get(t.id, 0.0)
        if abs(t.allocation - before) > tolerance:
            affected.append(t.id)

    # 5. Emit one ALLOCATION_SHIFTED per affected tendency, if a recorder
    #    was supplied.
    if recorder is not None:
        for tid in affected:
            t = new_state.tendencies.get(tid)
            recorder.emit_allocation_shift(
                origin_id=tid,
                new_allocation=t.allocation,
                delta=t.allocation - pre_alloc.get(tid, 0.0),
            )

    return ReseedResult(
        state=new_state,
        iterations=iterations,
        converged=converged,
        final_max_delta=final_delta,
        affected_ids=affected,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_substitutions(
    state: PresentState,
    substitutions: list[Substitution],
) -> dict[str, float]:
    """Apply each substitution in order. Returns the implied target allocations.

    The caller's specified ``new_tendency.allocation`` becomes the *target*
    for calibration -- the value the tendency should land on at the end of
    the reseed -- but it is *not* written directly into the live tendency
    set. Calibration drives the live allocation toward that target. This
    keeps the calibration loop the only thing that moves allocations,
    regardless of whether a substitution implies a small or large shift.

    Removals drop the tendency from the set, the lineage dict, and the
    graph. Additions inject a new tendency at allocation 0; calibration
    raises it. Replacements preserve the existing live allocation as the
    starting point so calibration has work to do.
    """
    targets: dict[str, float] = {
        t.id: t.allocation for t in state.tendencies.all()
    }

    for sub in substitutions:
        if sub.is_removal:
            state.tendencies.remove(sub.id)
            state.lineages.pop(sub.id, None)
            state.graph.remove_tendency(sub.id)
            targets.pop(sub.id, None)
            continue

        # Addition or replacement
        new_t = copy.deepcopy(sub.new_tendency)
        target_alloc = new_t.allocation

        if state.tendencies.has(new_t.id):
            # Replacement: keep the live allocation as the starting point so
            # calibration drives it toward the new target.
            existing = state.tendencies.get(new_t.id)
            new_t.allocation = existing.allocation
        else:
            # Addition: start at 0 and let calibration raise it.
            new_t.allocation = 0.0

        state.tendencies.add(new_t)
        state.lineages.setdefault(new_t.id, Lineage())
        state.graph.add_tendency(new_t.id)

        if sub.edges is not None:
            # Replace neighborhood: drop existing edges, install new ones.
            state.graph.remove_tendency(new_t.id)
            state.graph.add_tendency(new_t.id)
            for other_id, w in sub.edges.items():
                state.graph.add_edge(new_t.id, other_id, w)

        targets[new_t.id] = target_alloc

    # Normalize targets so they sum to 1 (caller may not have done this).
    total = sum(max(0.0, v) for v in targets.values())
    if total > 0:
        targets = {k: max(0.0, v) / total for k, v in targets.items()}
    elif targets:
        n = len(targets)
        targets = {k: 1.0 / n for k in targets}

    return targets


def _calibrate(
    tendencies: TendencySet,
    target_allocs: dict[str, float],
    *,
    tolerance: float,
    max_iterations: int,
    learning_rate: float,
    propagate_via_graph: bool = False,
    graph: Optional[StakeWeightGraph] = None,
    anchored_ids: Optional[set[str]] = None,
    novelty_modulated: bool = True,
    sharpen_among_ids: Optional[set[str]] = None,
    sharpen_strength: float = 1.0,
) -> tuple[int, bool, float]:
    """Iterate the contraction map until allocations settle.

    Two modes:

      Direct (default): each tendency's target is taken from
      ``target_allocs``; the loop blends current toward target.

      Graph-propagation: anchored tendencies (those the caller
      substituted) have explicit targets in ``target_allocs``.
      Non-anchored tendencies recompute their target each iteration
      based on their neighbors' allocations.

    Within graph-propagation, two sub-modes:

      Novelty-modulated (default): each tendency T uses its OWN edges
      as a reference frame -- its edge weight to a neighbor N is read
      as "the allocation T expects N to have." When N's actual
      allocation deviates from this expectation, the deviation is the
      novelty T experiences. Neighbors with high novelty contribute
      more weight to T's target update; expected neighbors contribute
      less. This is the engine's first-class attention mechanism: the
      receiver attends to surprises from its own reference frame.

      Plain weighted average: simple Laplace relaxation. Each
      neighbor's current allocation is averaged according to edge
      weight, ignoring whether that allocation was expected. This is
      the smoothed-out version, kept for diagnostics and comparison.

    Returns (iterations_run, converged?, final_max_delta).
    """
    if propagate_via_graph and (graph is None or anchored_ids is None):
        raise ValueError(
            "propagate_via_graph=True requires graph and anchored_ids"
        )

    last_max_delta = 0.0
    for it in range(1, max_iterations + 1):
        if propagate_via_graph:
            current = {t.id: t.allocation for t in tendencies.all()}
            iter_targets = dict(target_allocs)
            for t in tendencies.all():
                if t.id in anchored_ids:
                    continue
                neighbors = graph.weights.get(t.id, {})
                if not neighbors:
                    iter_targets[t.id] = t.allocation
                    continue

                if novelty_modulated:
                    # T's reference frame for what N "should" be is
                    # T's edge weight to N, normalized within T's
                    # neighborhood. Deviation = how surprising N's
                    # actual allocation is to T.
                    edge_total = sum(neighbors.values())
                    if edge_total <= 0:
                        iter_targets[t.id] = t.allocation
                        continue
                    expected = {
                        n_id: w / edge_total
                        for n_id, w in neighbors.items()
                    }
                    # Average neighbor expected allocation, in
                    # *neighbor* units: scale so a typical neighbor
                    # holds 1/n_neighbors of mass.
                    n_neighbors = len(neighbors)
                    typical_neighbor_mass = 1.0 / n_neighbors

                    novelty_scores: dict[str, float] = {}
                    for n_id in neighbors:
                        exp_alloc = expected[n_id] * typical_neighbor_mass * n_neighbors
                        # "expected" sums to 1; rescale so the typical
                        # expectation matches the global per-tendency
                        # average. Then deviation is allocation -
                        # expected.
                        actual = current.get(n_id, 0.0)
                        # Use absolute deviation as novelty; near-zero
                        # for as-expected, large for surprising.
                        novelty_scores[n_id] = abs(actual - exp_alloc)

                    # Combine edge weight (how much this tendency
                    # cares about this neighbor in principle) with
                    # novelty (how much actually-happening signal
                    # there is to attend to this iteration).
                    # Floor on novelty so even fully-expected
                    # neighbors contribute something.
                    nov_floor = 0.05
                    effective_weights = {
                        n_id: neighbors[n_id] * (nov_floor + novelty_scores[n_id])
                        for n_id in neighbors
                    }
                    eff_total = sum(effective_weights.values())
                    if eff_total <= 0:
                        iter_targets[t.id] = t.allocation
                        continue
                    weighted_sum = sum(
                        current.get(n_id, 0.0) * w
                        for n_id, w in effective_weights.items()
                    )
                    iter_targets[t.id] = weighted_sum / eff_total
                else:
                    total_weight = sum(neighbors.values())
                    if total_weight <= 0:
                        iter_targets[t.id] = t.allocation
                        continue
                    weighted_sum = sum(
                        current.get(n_id, 0.0) * w
                        for n_id, w in neighbors.items()
                    )
                    iter_targets[t.id] = weighted_sum / total_weight

            t_total = sum(max(0.0, v) for v in iter_targets.values())
            if t_total > 0:
                iter_targets = {k: max(0.0, v) / t_total for k, v in iter_targets.items()}
        else:
            iter_targets = target_allocs

        max_delta = 0.0
        for t in tendencies.all():
            target = iter_targets.get(t.id, t.allocation)
            delta = (target - t.allocation) * learning_rate
            t.allocation += delta
            if t.allocation < 0.0:
                t.allocation = 0.0
            if abs(delta) > max_delta:
                max_delta = abs(delta)
        tendencies.normalize()

        # Optional: sharpen competitive subset via softmax. This is the
        # discriminative-dynamics primitive the graph-walk averaging lacks.
        if sharpen_among_ids:
            _apply_softmax_sharpen(tendencies, sharpen_among_ids, sharpen_strength)

        last_max_delta = max_delta
        if max_delta < tolerance:
            return it, True, max_delta
    return max_iterations, False, last_max_delta


def _apply_softmax_sharpen(
    tendencies: TendencySet,
    sharpen_ids: set[str],
    strength: float,
) -> None:
    """Apply a softmax-style winner amplification within a subset.

    The total mass held by the sharpen subset is preserved: we just
    redistribute it according to softmax(strength * allocation). The
    rest of the tendencies are untouched. After redistribution, the
    full TendencySet is renormalized to sum to 1.

    strength=0 means uniform within the subset. strength=1 means
    standard softmax. Larger means more aggressive winner amplification.
    """
    import math
    members = [t for t in tendencies.all() if t.id in sharpen_ids]
    if len(members) < 2:
        return
    subset_total = sum(t.allocation for t in members)
    if subset_total <= 0:
        return

    # Scale by strength then softmax. Subtract the max for numerical stability.
    scaled = [strength * t.allocation for t in members]
    max_scaled = max(scaled)
    exp_scaled = [math.exp(s - max_scaled) for s in scaled]
    exp_sum = sum(exp_scaled)
    if exp_sum <= 0:
        return

    # Redistribute the existing subset_total according to softmax weights
    for t, e in zip(members, exp_scaled):
        t.allocation = subset_total * (e / exp_sum)

    tendencies.normalize()
