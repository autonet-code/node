"""Per-subtree PCA projection for the substrate visualization.

Validates ``WorldService.compute_subtree_projection``:

  (a) Only descendants of the requested parent_id appear in items.
  (b) Every returned item carries ``topic_xy`` and ``cluster_id``.
  (c) Calling twice with the same parent_id reuses the cache —
      same cluster_ids, same topic_xy values.
  (d) An unknown parent_id returns an empty item list.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nodes.common.world_service import WorldService
from world_model.generalized import Observation


def _coord(charter, embed_idx, mag, embedding_dim=1024):
    out = list(charter) + [0.0] * embedding_dim
    out[len(charter) + embed_idx] = mag
    return tuple(out)


def _obs(label, charter=(0.0, 0.0, 0.6, 0.0), embed_idx=0, mag=0.5):
    return Observation(
        id=f"obs_{label}",
        coords=_coord(charter, embed_idx, mag),
        label=label,
    )


def _seed_subtree(svc: WorldService) -> str:
    """Submit a handful of observations that sprout under the
    ``promotion_of_intelligence`` charter root (charter=(0,0,0.6,0)
    points there). Returns that root's node id, which is the
    ``parent_id`` callers should drill into.
    """
    for i in range(5):
        svc.submit_observation(
            _obs(f"x{i}", embed_idx=i, mag=0.3 + 0.1 * i),
            agent_id="A",
            sprout_rootless=True,
        )
    return svc._world.tendencies["promotion_of_intelligence"].tree.root_node.id


def test_only_descendants_appear_in_items(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_subtree_a", data_root=tmp_path)
    try:
        parent_id = _seed_subtree(svc)
        with svc._lock:
            descendants = svc._descendants_of_locked(parent_id)
        assert descendants, "fixture must seed at least one descendant"

        result = svc.compute_subtree_projection(parent_id)
        assert result["mode"] == "nodes"
        assert result["parent_id"] == parent_id
        assert result["items"], "expected at least one item"
        for item in result["items"]:
            assert item["node_id"] in descendants
    finally:
        svc.shutdown()


def test_every_item_has_topic_xy_and_cluster_id(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_subtree_b", data_root=tmp_path)
    try:
        parent_id = _seed_subtree(svc)
        result = svc.compute_subtree_projection(parent_id)
        assert result["items"]
        for item in result["items"]:
            assert "topic_xy" in item
            assert isinstance(item["topic_xy"], list) and len(item["topic_xy"]) == 2
            assert "cluster_id" in item
            assert isinstance(item["cluster_id"], str)
    finally:
        svc.shutdown()


def test_repeated_calls_cache_stable(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_subtree_c", data_root=tmp_path)
    try:
        parent_id = _seed_subtree(svc)
        first = svc.compute_subtree_projection(parent_id)
        second = svc.compute_subtree_projection(parent_id)
        # The cache returns the same dict instance, so identity holds.
        assert first is second
        # Cluster ids match across calls.
        ids_first = {(i["node_id"], i["tendency_id"]): i["cluster_id"] for i in first["items"]}
        ids_second = {(i["node_id"], i["tendency_id"]): i["cluster_id"] for i in second["items"]}
        assert ids_first == ids_second
    finally:
        svc.shutdown()


def test_unknown_parent_returns_empty(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_subtree_d", data_root=tmp_path)
    try:
        _seed_subtree(svc)
        result = svc.compute_subtree_projection("does-not-exist-xyz")
        assert result["mode"] == "nodes"
        assert result["items"] == []
        assert result["parent_id"] == "does-not-exist-xyz"
    finally:
        svc.shutdown()


def test_other_charter_root_has_no_descendants(tmp_path: Path):
    """All seeded sprouts cluster under promotion_of_intelligence;
    asking for the life_precious root yields nothing. Confirms the
    parent_id filter is doing real work, not just letting everything
    through."""
    svc = WorldService(rpb_address="rpb_subtree_e", data_root=tmp_path)
    try:
        _seed_subtree(svc)
        life_root = svc._world.tendencies["life_precious"].tree.root_node.id
        result = svc.compute_subtree_projection(life_root)
        assert result["items"] == []
    finally:
        svc.shutdown()
