"""Phase 6.1: transient probe API on WorldService.

The probe reads the persistent world (locate + render) without
mutating it. Tests verify:

  1. Probing returns a sensible region for a query close to seeded
     nodes; charter-only worlds return a small but non-empty region
     (the four charter roots).
  2. Probing does NOT mutate the world: node count unchanged, root
     scores unchanged, no entries appear in events.jsonl, no events
     accumulate in the open epoch buffer.
  3. Distance ranking: a query close to one seeded region should
     rank that region's nodes above unrelated regions.
  4. Modes: general mode works; alignment mode is gracefully
     deferred (raises NotImplementedError).
  5. Empty world: probing a charter-only world returns the charter
     roots as the region.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from nodes.common.world_service import WorldService
from world_model.generalized import Observation


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _seed_world(tmp_path: Path, rpb: str) -> WorldService:
    """Build a WorldService with three usefulness regions seeded at
    distinct embedding-tail positions, plus charter contributions."""
    svc = WorldService(rpb_address=rpb, data_root=tmp_path)
    seeds = [
        ("intel_a", _coord((0.0, 0.0, 0.6, 0.0), embed_idx=10, mag=0.7)),
        ("intel_b", _coord((0.0, 0.0, 0.6, 0.0), embed_idx=12, mag=0.6)),
        ("evol_a",  _coord((0.0, 0.0, 0.0, 0.6), embed_idx=400, mag=0.7)),
        ("evol_b",  _coord((0.0, 0.0, 0.0, 0.6), embed_idx=402, mag=0.6)),
        ("life_a",  _coord((0.6, 0.0, 0.0, 0.0), embed_idx=800, mag=0.7)),
    ]
    for label, c in seeds:
        svc.submit_observation(
            Observation(id=f"obs_{label}", coords=c, label=label),
            agent_id="seeder",
            sprout_under_charter=True,
            sprout_rootless=True,
        )
    return svc


# ---------------------------------------------------------------------------
# Returns + non-mutation
# ---------------------------------------------------------------------------


def test_probe_returns_region_for_seeded_query(tmp_path: Path):
    svc = _seed_world(tmp_path, "rpb_probe_a")
    try:
        # Query close to the intelligence-axis seeds.
        q = _coord((0.0, 0.0, 0.6, 0.0), embed_idx=11, mag=0.65)
        result = svc.probe_inference(q, max_results=8)
        assert result["mode"] == "general"
        assert result["n_results"] > 0
        # Region carries the expected fields.
        first = result["region"][0]
        assert "node_id" in first and "distance" in first and "score" in first
        # Render output is structured.
        assert isinstance(result["render"], dict)
    finally:
        svc.shutdown()


def test_probe_does_not_mutate_world(tmp_path: Path):
    """The signature property: a probe is read-only."""
    svc = _seed_world(tmp_path, "rpb_probe_b")
    try:
        nodes_before = svc.stats()["n_nodes"]
        scores_before = dict(svc.read_root_scores())
        events_applied_before = svc._events_applied

        q = _coord((0.0, 0.0, 0.6, 0.0), embed_idx=11, mag=0.65)
        svc.probe_inference(q)
        svc.probe_inference(q)
        svc.probe_inference(q)

        assert svc.stats()["n_nodes"] == nodes_before
        assert svc.read_root_scores() == scores_before
        assert svc._events_applied == events_applied_before
    finally:
        svc.shutdown()


def test_probe_does_not_pollute_event_log(tmp_path: Path):
    """Probes must NOT write to events.jsonl. That log is for state-
    changing events only; probes are reads."""
    rpb = "rpb_probe_c"
    svc = _seed_world(tmp_path, rpb)
    try:
        events_log = tmp_path / rpb / "events.jsonl"
        bytes_before = events_log.stat().st_size
        q = _coord((0.0, 0.0, 0.6, 0.0), embed_idx=11, mag=0.65)
        svc.probe_inference(q)
        bytes_after = events_log.stat().st_size
        assert bytes_after == bytes_before, (
            f"events.jsonl grew during probe: {bytes_before} -> {bytes_after}"
        )
    finally:
        svc.shutdown()


def test_probe_does_not_pollute_open_epoch_buffer(tmp_path: Path):
    """If an epoch is open during a probe, the probe must NOT add
    events to the attribution buffer. Probes don't earn mint."""
    svc = _seed_world(tmp_path, "rpb_probe_d")
    try:
        svc.open_epoch("e_during_probe")
        events_in_epoch_before = len(svc._epoch_events)
        q = _coord((0.0, 0.0, 0.6, 0.0), embed_idx=11, mag=0.65)
        svc.probe_inference(q)
        svc.probe_inference(q)
        assert len(svc._epoch_events) == events_in_epoch_before
        svc.close_epoch()
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Distance ranking
# ---------------------------------------------------------------------------


def test_probe_ranks_closer_seeds_higher(tmp_path: Path):
    """A query near intel-axis seeds should rank intel-axis nodes
    above unrelated evolution-axis nodes."""
    svc = _seed_world(tmp_path, "rpb_probe_e")
    try:
        q = _coord((0.0, 0.0, 0.6, 0.0), embed_idx=11, mag=0.65)
        result = svc.probe_inference(q, max_results=20)

        labels = [r["label"] for r in result["region"]]
        # The intel seeds should appear before any of the evol_a /
        # evol_b / life_a labels.
        intel_positions = [
            i for i, l in enumerate(labels) if "intel" in l
        ]
        evol_positions = [
            i for i, l in enumerate(labels) if "evol" in l
        ]
        assert intel_positions, f"no intel labels found in {labels}"
        if evol_positions:
            assert min(intel_positions) < min(evol_positions), (
                f"intel didn't rank above evol: {labels}"
            )
    finally:
        svc.shutdown()


def test_probe_max_results_caps_region_size(tmp_path: Path):
    svc = _seed_world(tmp_path, "rpb_probe_f")
    try:
        q = _coord((0.0, 0.0, 0.6, 0.0), embed_idx=11, mag=0.65)
        small = svc.probe_inference(q, max_results=2)
        assert small["n_results"] <= 2
        # A larger cap returns more (subject to the world having more).
        big = svc.probe_inference(q, max_results=50)
        assert big["n_results"] >= small["n_results"]
    finally:
        svc.shutdown()


def test_probe_max_distance_filters_far_nodes(tmp_path: Path):
    """A tight max_distance excludes far nodes."""
    svc = _seed_world(tmp_path, "rpb_probe_g")
    try:
        q = _coord((0.0, 0.0, 0.6, 0.0), embed_idx=11, mag=0.65)
        unbounded = svc.probe_inference(q, max_results=50)
        bounded = svc.probe_inference(q, max_results=50, max_distance=0.3)
        assert bounded["n_results"] <= unbounded["n_results"]
        # Every returned node is within the distance bound.
        for r in bounded["region"]:
            assert r["distance"] <= 0.3 + 1e-9
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def test_alignment_mode_raises_not_implemented(tmp_path: Path):
    """Phase 6.1 ships general mode only. Alignment is deferred."""
    svc = _seed_world(tmp_path, "rpb_probe_h")
    try:
        q = _coord((0.0, 0.0, 0.6, 0.0), embed_idx=11, mag=0.65)
        with pytest.raises(NotImplementedError):
            svc.probe_inference(q, mode="alignment")
    finally:
        svc.shutdown()


def test_unknown_mode_raises_value_error(tmp_path: Path):
    svc = _seed_world(tmp_path, "rpb_probe_i")
    try:
        q = _coord((0.0, 0.0, 0.6, 0.0), embed_idx=11, mag=0.65)
        with pytest.raises(ValueError):
            svc.probe_inference(q, mode="totally-fake-mode")
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Empty world
# ---------------------------------------------------------------------------


def test_probe_on_charter_only_world_returns_charter_roots(tmp_path: Path):
    """A fresh WorldService has only the four charter roots. A probe
    against any coords should return some subset of them, not crash
    or return zero results."""
    svc = WorldService(rpb_address="rpb_probe_empty", data_root=tmp_path)
    try:
        q = _coord((0.5, 0.0, 0.0, 0.0), embed_idx=0, mag=0.0)
        result = svc.probe_inference(q, max_results=10)
        assert result["mode"] == "general"
        # Charter has 4 roots; probe should pick them up.
        assert result["n_results"] >= 1
    finally:
        svc.shutdown()
