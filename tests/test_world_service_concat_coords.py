"""Smoke test for charter+usefulness coexistence in one world.

Validates the design point that charter roots (N_DIMS axes) and rootless
usefulness nodes (high-D semantic embedding) can live in one world
when coordinates are concatenated as ``[charter_Nd | embedding_Md]``.

Two assertions
--------------

  1. **Charter scoring updates**: an observation with non-zero charter
     projection actually moves charter root scores. This is the
     "agent activity influences alignment scoring" path.
  2. **Locator retrieval works on the embedding tail**: usefulness
     nodes anchored deep in the embedding tail (zero on charter axes)
     are findable by querying with a similar embedding tail, but NOT
     pulled in by a query that only looks like a charter axis. This
     validates that distance properly accounts for the full vector.

Padding is required: charter anchors must be padded with zeros in the
embedding tail. Otherwise ``_euclid``'s ``min(len(a), len(b))``
truncation makes any embedding-only node look right next to charter
origin (false locality match).
"""

from __future__ import annotations

from typing import Tuple

import pytest

from nodes.common.world_model_substrate.adapter import N_DIMS
from nodes.common.world_service import WorldService
from world_model.generalized import Observation, default_locator, equilibrate
from world_model.models.tree import Position


def _coord(charter: Tuple[float, ...], embed_tail: Tuple[float, ...]) -> Tuple[float, ...]:
    """Build a coord vector. ``charter`` may be shorter than N_DIMS; it
    is zero-padded so tests written for the 4-axis charter keep working
    when N_DIMS grows.
    """
    head = tuple(charter)
    if len(head) < N_DIMS:
        head = head + (0.0,) * (N_DIMS - len(head))
    return head + tuple(embed_tail)


def test_charter_anchors_padded_to_full_dim(tmp_path):
    """Smoke test 0: WorldService built with embedding_dim=N has charter
    anchors of length 4+N (the embedding tail is zero-padded)."""
    embed_dim = 8
    svc = WorldService(
        rpb_address="rpb_concat_pad",
        data_root=tmp_path,
        embedding_dim=embed_dim,
    )
    try:
        for tendency in svc._world.tendencies.values():
            assert len(tendency.anchor) == N_DIMS + embed_dim, (
                f"tendency {tendency.id} has anchor of length "
                f"{len(tendency.anchor)}, expected {N_DIMS + embed_dim}"
            )
            # Embedding tail must be zeros; only the charter head is nonzero.
            tail = tendency.anchor[N_DIMS:]
            assert all(t == 0.0 for t in tail), \
                f"tendency {tendency.id} has nonzero embedding tail: {tail}"
    finally:
        svc.shutdown()


def test_charter_scoring_updates_under_concatenated_coords(tmp_path):
    """Smoke test 1: an observation with charter projection on an axis
    moves that axis's root score, even when the observation also has a
    rich embedding tail."""
    embed_dim = 16
    svc = WorldService(
        rpb_address="rpb_concat_charter",
        data_root=tmp_path,
        embedding_dim=embed_dim,
    )
    try:
        before = svc.read_root_scores()
        # Observation strongly oriented toward "promotion_of_intelligence"
        # (axis 2) with arbitrary embedding tail.
        obs_coord = _coord(
            charter=(0.1, 0.1, 0.9, 0.1),
            embed_tail=tuple(0.05 * i for i in range(embed_dim)),
        )
        obs = Observation(id="obs_intel_1", coords=obs_coord, label="intel-witness")

        # Use submit_observation rather than submit_events so we mimic
        # the path Phase 2 will take when handed work units. The
        # service applies the observation, equilibrates, persists.
        svc.submit_observation(obs, agent_id="smoketest")

        after = svc.read_root_scores()

        # The intelligence axis must move (substantially more than the
        # other axes, which got only weak weight from the same obs).
        delta = {k: after[k] - before[k] for k in before}
        assert delta["promotion_of_intelligence"] > 0, \
            f"intelligence axis didn't move: delta={delta}"
        # And the embedding tail shouldn't have absorbed all the
        # signal — the charter head must still drive scoring.
        assert delta["promotion_of_intelligence"] >= max(
            delta["life_precious"],
            delta["self_preservation"],
            delta["evolution"],
        ), f"intelligence didn't dominate: {delta}"
    finally:
        svc.shutdown()


def test_locator_retrieves_on_embedding_tail(tmp_path):
    """Smoke test 2: usefulness nodes anchored only in the embedding
    tail (zero charter head) must be findable by an embedding-tail
    query. They must NOT be wrongly retrieved by a charter-only query
    — that's the bug that occurs without padding."""
    embed_dim = 8
    svc = WorldService(
        rpb_address="rpb_concat_locator",
        data_root=tmp_path,
        embedding_dim=embed_dim,
    )
    try:
        # Drop two usefulness nodes deep in the embedding tail, far
        # from each other. Both have zero charter projection.
        a_tail = tuple([1.0] + [0.0] * (embed_dim - 1))     # along e1
        b_tail = tuple([0.0] * (embed_dim - 1) + [1.0])     # along eN
        a_coord = _coord(charter=(0.0, 0.0, 0.0, 0.0), embed_tail=a_tail)
        b_coord = _coord(charter=(0.0, 0.0, 0.0, 0.0), embed_tail=b_tail)

        svc.submit_observation(
            Observation(id="obs_useA", coords=a_coord, label="useful_A"),
            agent_id="smoketest",
            sprout_under_charter=False,
            sprout_rootless=True,
        )
        svc.submit_observation(
            Observation(id="obs_useB", coords=b_coord, label="useful_B"),
            agent_id="smoketest",
            sprout_under_charter=False,
            sprout_rootless=True,
        )

        # Query close to a_tail. Should find A, not B.
        q_a = _coord(charter=(0.0, 0.0, 0.0, 0.0), embed_tail=tuple([0.95] + [0.0] * (embed_dim - 1)))
        results_a = svc.locate(q_a, max_results=2, max_distance=0.5)
        labels_a = {r["label"] for r in results_a}
        assert "useful_A" in labels_a, f"expected useful_A in {labels_a}"
        assert "useful_B" not in labels_a, f"didn't expect useful_B in {labels_a}"

        # A pure-charter query (no embedding signal) MUST NOT pick up
        # the usefulness nodes. Without padding charter anchors, this
        # is the failure mode: usefulness nodes look "right at the
        # charter origin" because dim mismatch silently truncates.
        q_charter = _coord(charter=(0.5, 0.0, 0.0, 0.0), embed_tail=(0.0,) * embed_dim)
        results_charter = svc.locate(q_charter, max_results=4, max_distance=0.4)
        charter_labels = {r["label"] for r in results_charter}
        assert "useful_A" not in charter_labels, \
            f"usefulness node leaked into charter-only query: {charter_labels}"
        assert "useful_B" not in charter_labels, \
            f"usefulness node leaked into charter-only query: {charter_labels}"
    finally:
        svc.shutdown()


def test_default_embedding_dim_is_full_usefulness_layer(tmp_path):
    """The daemon defaults to embedding_dim=1024 so the usefulness
    layer is on out of the box. Pure-charter mode is opt-in via
    embedding_dim=0."""
    svc = WorldService(rpb_address="rpb_concat_default", data_root=tmp_path)
    try:
        for tendency in svc._world.tendencies.values():
            assert len(tendency.anchor) == N_DIMS + 1024
    finally:
        svc.shutdown()


def test_pure_charter_mode_via_embedding_dim_zero(tmp_path):
    """Setting embedding_dim=0 explicitly returns the charter-only
    world (back-compat path for callers who want it)."""
    svc = WorldService(
        rpb_address="rpb_concat_pure_charter",
        data_root=tmp_path,
        embedding_dim=0,
    )
    try:
        for tendency in svc._world.tendencies.values():
            assert len(tendency.anchor) == N_DIMS
    finally:
        svc.shutdown()
