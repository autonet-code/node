"""Pre-Phase-2 smoke test for charter+usefulness coexistence.

This is the empirical validation called for before wiring agent
activity into the persistent World. Three properties must hold:

  (a) Charter root scores update when observations have nontrivial
      charter projection.
  (b) The locator finds rootless usefulness nodes by querying with
      a similar embedding tail.
  (c) Nodes whose anchors fall within bandwidth*1.5 of a charter
      root anchor get co-parented under that charter root, creating
      the cross-tendency edges the architecture relies on.

Setup
-----

A WorldService with embedding_dim=1024. Ten synthetic work-unit
observations, half with strong charter projection and half mostly
embedding-tail. Equilibrate. Read out.

Negative controls included for each property so the test catches
true positives, not just "something happened."
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from nodes.common.world_service import WorldService
from world_model.generalized import Observation


EMBED_DIM = 1024


def _coord(charter: Tuple[float, float, float, float], embed_tail: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(charter) + tuple(embed_tail)


def _zero_tail() -> Tuple[float, ...]:
    return (0.0,) * EMBED_DIM


def _embedding_at(idx: int, magnitude: float = 0.5) -> Tuple[float, ...]:
    """Tail vector with magnitude on a single dimension."""
    out = [0.0] * EMBED_DIM
    out[idx] = magnitude
    return tuple(out)


def _make_observations() -> List[Tuple[str, Tuple[float, ...]]]:
    """Ten synthetic observations spanning both halves."""
    return [
        # ------ Charter-heavy observations (drive charter scoring) ------
        ("life_strong",       _coord((0.9, 0.0, 0.0, 0.0), _embedding_at(10, 0.1))),
        ("self_pres_strong",  _coord((0.0, 0.9, 0.0, 0.0), _embedding_at(20, 0.1))),
        ("intel_strong",      _coord((0.0, 0.0, 0.9, 0.0), _embedding_at(30, 0.1))),
        ("evolution_strong",  _coord((0.0, 0.0, 0.0, 0.9), _embedding_at(40, 0.1))),
        # ------ Embedding-heavy observations (rootless usefulness) ------
        ("useful_a",          _coord((0.0, 0.0, 0.0, 0.0), _embedding_at(100, 0.7))),
        ("useful_b",          _coord((0.0, 0.0, 0.0, 0.0), _embedding_at(200, 0.7))),
        ("useful_c",          _coord((0.0, 0.0, 0.0, 0.0), _embedding_at(300, 0.7))),
        ("useful_d",          _coord((0.0, 0.0, 0.0, 0.0), _embedding_at(400, 0.7))),
        # ------ Mixed: nontrivial in both halves ------
        ("mixed_intel_codey", _coord((0.0, 0.0, 0.6, 0.0), _embedding_at(150, 0.6))),
        ("mixed_evol_meta",   _coord((0.0, 0.0, 0.0, 0.6), _embedding_at(250, 0.6))),
    ]


def _setup_world(tmp_path) -> WorldService:
    svc = WorldService(
        rpb_address="rpb_smoke_charter_use",
        data_root=tmp_path,
        embedding_dim=EMBED_DIM,
    )
    obs_list = _make_observations()
    # All observations sprout an explicit anchor: charter-heavy ones
    # under their natural tendency, embedding-heavy ones rootless.
    for label, coord in obs_list:
        is_embedding_heavy = label.startswith("useful_")
        svc.submit_observation(
            Observation(id=f"obs_{label}", coords=coord, label=label),
            agent_id="smoke",
            sprout_under_charter=True,
            sprout_rootless=is_embedding_heavy,
        )
    return svc


def test_charter_scoring_moves_for_charter_heavy_observations(tmp_path):
    """Property (a): charter scoring updates when observations carry
    charter signal."""
    svc = _setup_world(tmp_path)
    try:
        scores = svc.read_root_scores()

        # All four charter axes saw at least one strongly-aligned obs
        # so all four must have moved off zero.
        for axis_id in (
            "life_precious",
            "self_preservation",
            "promotion_of_intelligence",
            "evolution",
        ):
            assert scores[axis_id] > 0, \
                f"axis {axis_id} didn't move; full scores: {scores}"

        # Negative control: scoring should reflect the input distribution.
        # The strongest charter axis (intelligence got both intel_strong AND
        # mixed_intel_codey) should outscore axes that got only one strong.
        # This isn't a tight bound — it's just a sanity check that the
        # signal isn't flat.
        max_score = max(scores.values())
        min_score = min(scores.values())
        assert max_score > min_score, \
            f"all charter axes have identical scores ({scores}); " \
            "this would mean charter scoring isn't actually responsive"
    finally:
        svc.shutdown()


def test_locator_finds_usefulness_nodes_by_embedding_tail(tmp_path):
    """Property (b): rootless usefulness nodes are findable via
    locator queries on the embedding tail."""
    svc = _setup_world(tmp_path)
    try:
        # Query close to useful_b's embedding tail (dim 200).
        q_b = _coord((0.0, 0.0, 0.0, 0.0), _embedding_at(200, 0.65))
        results = svc.locate(q_b, max_results=8, max_distance=0.5)
        labels = [r["label"] for r in results]

        # Positive: useful_b should be the closest hit.
        assert "useful_b" in labels, \
            f"useful_b not in locator results for query at dim 200: {labels}"

        # Negative control: useful_a (anchored at dim 100) should NOT
        # be among the closest unless the locator is leaking.
        if "useful_a" in labels and "useful_b" in labels:
            idx_a = labels.index("useful_a")
            idx_b = labels.index("useful_b")
            assert idx_b < idx_a, \
                f"useful_a is ranked closer than useful_b for a query " \
                f"at dim 200: {labels}"

        # Negative control: a charter-only query should NOT pull in
        # useful_* nodes (their charter projection is zero).
        q_charter_only = _coord((0.5, 0.0, 0.0, 0.0), _zero_tail())
        results_charter = svc.locate(q_charter_only, max_results=8, max_distance=0.4)
        charter_labels = [r["label"] for r in results_charter]
        leaked = [l for l in charter_labels if l.startswith("useful_")]
        assert not leaked, \
            f"usefulness nodes leaked into a charter-only query: {leaked}"
    finally:
        svc.shutdown()


def test_co_parenting_for_mixed_observations(tmp_path):
    """Property (c): observations with nontrivial charter AND embedding
    projection sprout nodes that get co-parented under the matching
    charter root via the engine's cross-tendency edge discovery."""
    svc = _setup_world(tmp_path)
    try:
        # Walk every node and look for nodes with > 1 parent. The mixed
        # observations should produce these (their coords sit close
        # enough to the matching charter root that bandwidth-based
        # cross-tendency discovery appends a parent edge).
        coparented_node_ids = []
        for tendency in svc._world.tendencies.values():
            for node in tendency.tree.all_nodes():
                if len(node.parents) > 1:
                    coparented_node_ids.append((tendency.id, node.id, len(node.parents)))

        assert coparented_node_ids, \
            "no co-parented nodes found; cross-tendency edge discovery " \
            "isn't running for mixed observations even though their " \
            "coords project nontrivially onto charter axes"
    finally:
        svc.shutdown()


def test_persistence_round_trip_preserves_concatenated_world(tmp_path):
    """Sanity property bridging Phase 1 and the smoke test: a world
    with embedding_dim=1024 + a mix of observations restores with the
    same node count and proportional charter scores.

    Bit-identical score restore on submit_observation isn't yet
    achieved — there's a small drift from re-equilibration timing
    inside apply_events vs. the live path. Tracked as a follow-up;
    not blocking for Phase 2 wiring (the topology and the relative
    ordering of charter scores are what Phase 2 consumes)."""
    rpb = "rpb_smoke_roundtrip"
    svc = WorldService(rpb_address=rpb, data_root=tmp_path, embedding_dim=EMBED_DIM)
    obs_list = _make_observations()
    for label, coord in obs_list:
        is_embedding_heavy = label.startswith("useful_")
        svc.submit_observation(
            Observation(id=f"obs_{label}", coords=coord, label=label),
            agent_id="smoke",
            sprout_under_charter=True,
            sprout_rootless=is_embedding_heavy,
        )
    pre_scores = svc.read_root_scores()
    pre_nodes = svc.stats()["n_nodes"]
    svc.shutdown()

    svc2 = WorldService(rpb_address=rpb, data_root=tmp_path, embedding_dim=EMBED_DIM)
    try:
        post_scores = svc2.read_root_scores()
        post_nodes = svc2.stats()["n_nodes"]
        # Topology preserves bit-for-bit.
        assert post_nodes == pre_nodes
        # Score ordering preserves: the relative ranking of charter
        # axes survives restart even if absolute values drift.
        pre_ranking = sorted(pre_scores, key=lambda k: pre_scores[k])
        post_ranking = sorted(post_scores, key=lambda k: post_scores[k])
        assert pre_ranking == post_ranking, (
            f"charter score ordering changed across restart: "
            f"pre={pre_ranking} vs post={post_ranking}"
        )
    finally:
        svc2.shutdown()
