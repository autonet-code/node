"""Phase 6.2: query → coords adapter on WorldService.

Validates that text queries become coord vectors compatible with the
substrate's [charter_N + embedding_dim] coordinate space.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nodes.common.world_model_substrate.adapter import N_DIMS
from nodes.common.world_service import WorldService


def test_coords_for_text_returns_full_dim(tmp_path: Path):
    """Length matches WorldService.embedding_dim + N_DIMS charter dims."""
    svc = WorldService(rpb_address="rpb_qct_a", data_root=tmp_path, embedding_dim=512)
    try:
        coords = svc.coords_for_text("how does authentication work in this codebase")
        assert len(coords) == N_DIMS + 512
    finally:
        svc.shutdown()


def test_coords_for_text_charter_head_zero_by_default(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_qct_b", data_root=tmp_path)
    try:
        coords = svc.coords_for_text("anything at all")
        for i in range(N_DIMS):
            assert coords[i] == 0.0
    finally:
        svc.shutdown()


def test_coords_for_text_charter_head_override(tmp_path: Path):
    """Caller can supply a non-zero charter head (Phase 6.4+ hook)."""
    svc = WorldService(rpb_address="rpb_qct_c", data_root=tmp_path)
    try:
        # Bias the intelligence axis (index 2 in the charter).
        head = [0.0] * N_DIMS
        head[2] = 0.7
        coords = svc.coords_for_text("test", charter_head=tuple(head))
        assert coords[2] == 0.7
        assert coords[0] == 0.0
        assert coords[3] == 0.0
    finally:
        svc.shutdown()


def test_coords_for_text_charter_head_wrong_length_rejected(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_qct_d", data_root=tmp_path)
    try:
        with pytest.raises(ValueError):
            svc.coords_for_text("test", charter_head=(0.5, 0.0, 0.0))
    finally:
        svc.shutdown()


def test_coords_for_text_deterministic_for_same_text(tmp_path: Path):
    """Same text + same embedder → same coords."""
    svc = WorldService(rpb_address="rpb_qct_e", data_root=tmp_path)
    try:
        a = svc.coords_for_text("the same query")
        b = svc.coords_for_text("the same query")
        assert a == b
    finally:
        svc.shutdown()


def test_coords_for_text_different_for_different_text(tmp_path: Path):
    """Distinct queries should embed differently (non-degenerate)."""
    svc = WorldService(rpb_address="rpb_qct_f", data_root=tmp_path)
    try:
        a = svc.coords_for_text("how does authentication work in this codebase")
        b = svc.coords_for_text("what is the meaning of life")
        # We can't predict exact values, but at least the embedding
        # tails should not be identical.
        assert a[N_DIMS:] != b[N_DIMS:], "embedder produced identical tails for different texts"
    finally:
        svc.shutdown()


def test_coords_for_text_works_with_zero_embedding_dim(tmp_path: Path):
    """Pure-charter mode (embedding_dim=0): coords are just the head."""
    svc = WorldService(
        rpb_address="rpb_qct_g", data_root=tmp_path, embedding_dim=0,
    )
    try:
        head = [0.0] * N_DIMS
        head[0] = 0.5
        coords = svc.coords_for_text("test", charter_head=tuple(head))
        assert len(coords) == N_DIMS
        assert coords == tuple(head)
    finally:
        svc.shutdown()


def test_coords_for_text_feeds_probe_inference(tmp_path: Path):
    """End-to-end: text query → coords → probe_inference. The pipe
    works; that's what 6.3/6.4 will lean on."""
    from world_model.generalized import Observation

    svc = WorldService(rpb_address="rpb_qct_h", data_root=tmp_path)
    # Seed something so the probe has nodes to find.
    seed_coords = svc.coords_for_text("authentication and login flow")
    svc.submit_observation(
        Observation(id="obs_seed", coords=seed_coords, label="auth_seed"),
        agent_id="seeder",
        sprout_under_charter=True,
        sprout_rootless=True,
    )

    try:
        q = svc.coords_for_text("how does authentication work")
        result = svc.probe_inference(q, max_results=8)
        assert result["mode"] == "general"
        assert result["n_results"] > 0
    finally:
        svc.shutdown()
