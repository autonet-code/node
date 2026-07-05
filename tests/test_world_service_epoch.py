"""Phase 3 tests: epoch reward windows + scheduler.

Validates:
  1. open_epoch / close_epoch lifecycle (empty epoch -> empty mint).
  2. Events submitted during the epoch are attributed to their agent
     as NOVELTY diagnostics (one earning rail: prose never mints).
  3. Multiple agents producing events at the same epoch get separate
     novelty attributions.
  4. The charter mint gate can be turned on/off via close_epoch's
     apply_gate flag.
  5. ``EpochScheduler.maybe_tick`` only fires when the wall-clock
     interval has elapsed; ``force_tick`` bypasses the clock.
  6. ONE EARNING RAIL: the local projection close mints through
     ``compute_tool_mint`` only (registration + attested receipts),
     matching the authoritative federated close; score movement stays
     a labeled diagnostic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from nodes.common.world_service import WorldService
from world_model.generalized import Observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _obs(label, charter=(0.0, 0.0, 0.0, 0.0), embed_idx=0, mag=0.5):
    return Observation(
        id=f"obs_{label}",
        coords=_coord(charter, embed_idx, mag),
        label=label,
    )


# Tool-rail fixtures (mirrors tests/test_tool_mint.py, sized to the
# service's embedding dim).
_TOOL_DIGEST = "ab" * 32
_TOOL_AUTHOR = "toolsmith"
_TOOL_EMBED_DIM = 64


def _tool_coords():
    from nodes.common.world_model_substrate.adapter import N_DIMS
    out = [0.0] * (N_DIMS + _TOOL_EMBED_DIM)
    out[4] = 0.8
    out[N_DIMS + 3] = 0.5
    return out


def _tool_registration_event(seq=1, digest=_TOOL_DIGEST, author=_TOOL_AUTHOR):
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": author,
        "tendency_id": "correctness",
        "parent_id": "solver_root",
        "node_id": f"tm_{digest[:12]}",
        "position": "pro",
        "coords": _tool_coords(),
        "polarity_axis": _tool_coords(),
        "content": f"tool {digest[:8]}: does a thing",
        "author_post": True,
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": "pinned", "author": author},
    }


def _tool_receipt_event(seq, caller, digest=_TOOL_DIGEST):
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": caller,
        "manifest_digest": digest,
        "tool_author": _TOOL_AUTHOR,
        "receipt_digest": f"r{seq:02d}" * 8,
        "ok": True,
        "fee_atn": 0.0,
        "attested": True,
        "score": 0.8,
    }


def _tool_vet_event(seq, vetter, digest=_TOOL_DIGEST):
    """Affirmative vet (greenlight input; not usage). Two distinct
    vetters reach VET_QUORUM."""
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": vetter,
        "manifest_digest": digest,
        "tool_author": _TOOL_AUTHOR,
        "receipt_digest": f"v{seq:02d}" * 8,
        "ok": True,
        "fee_atn": 0.0,
        "vet": True,
    }


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_open_close_with_no_events_yields_empty_mint(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_epoch_a", data_root=tmp_path)
    try:
        svc.open_epoch("epoch_zero")
        result = svc.close_epoch()
        assert result["epoch_id"] == "epoch_zero"
        assert result["agent_mint"] == {}
        assert result["total_mint"] == 0.0
        assert result["n_events"] == 0
    finally:
        svc.shutdown()


def test_close_without_open_is_safe(tmp_path: Path):
    """Closing an epoch when none is open is a no-op, not a crash."""
    svc = WorldService(rpb_address="rpb_epoch_b", data_root=tmp_path)
    try:
        result = svc.close_epoch()
        assert result["epoch_id"] is None
        assert result["agent_mint"] == {}
    finally:
        svc.shutdown()


def test_double_open_auto_closes_prior_epoch(tmp_path: Path):
    """Calling open_epoch while another is open auto-closes the
    prior one rather than corrupting state."""
    svc = WorldService(rpb_address="rpb_epoch_c", data_root=tmp_path)
    try:
        svc.open_epoch("first")
        svc.submit_observation(_obs("a", charter=(0.5, 0, 0, 0)), agent_id="A")
        svc.open_epoch("second")  # implicitly closes first
        # History should contain the auto-closed epoch.
        history = svc.epoch_history
        assert len(history) == 1
        assert history[0]["epoch_id"] == "first"
        # And second is now open.
        assert svc.current_epoch_id == "second"
    finally:
        svc.shutdown()


def test_epoch_buffers_events_submitted_within_window(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_epoch_d", data_root=tmp_path)
    try:
        # Pre-epoch event — should NOT be attributed to the epoch.
        svc.submit_observation(_obs("pre", charter=(0.5, 0, 0, 0)), agent_id="A")
        svc.open_epoch("e1")
        # In-epoch events.
        svc.submit_observation(_obs("intel", charter=(0.0, 0, 0.7, 0)), agent_id="A")
        svc.submit_observation(_obs("evol", charter=(0.0, 0, 0, 0.7)), agent_id="A")
        result = svc.close_epoch()
        # One earning rail: prose events never mint — attribution shows
        # up as novelty (diagnostic), and the score-movement per-agent
        # map stays in its labeled diagnostic slot.
        assert result["agent_mint"] == {}
        assert "A" in result["agent_novelty"], result
        assert result["agent_novelty"]["A"] > 0
        assert result["score_agent_mint"].get("A", 0) > 0
        # And the n_events recorded reflects the in-epoch events only,
        # which are causal events (observation_added + sprout for each
        # submit_observation).
        assert result["n_events"] >= 2
    finally:
        svc.shutdown()


def test_two_agents_get_separate_attribution(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_epoch_e", data_root=tmp_path)
    try:
        svc.open_epoch()
        # Agent A contributes intelligence-axis activity.
        svc.submit_observation(_obs("A_intel", charter=(0.0, 0, 0.8, 0), embed_idx=10), agent_id="A")
        svc.submit_observation(_obs("A_intel2", charter=(0.0, 0, 0.7, 0), embed_idx=11), agent_id="A")
        # Agent B contributes evolution-axis activity.
        svc.submit_observation(_obs("B_evol", charter=(0.0, 0, 0, 0.8), embed_idx=20), agent_id="B")
        result = svc.close_epoch()
        # Both agents must appear in agent_novelty with a positive
        # amount. (We don't assert magnitudes — those depend on
        # equilibration — but we do assert separation: the same agent
        # isn't credited for both.) agent_mint stays empty: prose
        # activity never mints on the tool-only rail.
        assert result["agent_mint"] == {}
        novelty = result["agent_novelty"]
        assert "A" in novelty, novelty
        assert "B" in novelty, novelty
        assert novelty["A"] > 0
        assert novelty["B"] > 0
    finally:
        svc.shutdown()


def test_apply_mint_gate_can_be_disabled(tmp_path: Path):
    """When apply_gate=False, raw mint is returned (no charter
    suppression). When True (default), the gate runs."""
    svc = WorldService(rpb_address="rpb_epoch_f", data_root=tmp_path)
    try:
        svc.open_epoch()
        svc.submit_observation(_obs("intel", charter=(0.0, 0, 0.9, 0)), agent_id="A")
        # No charter violations posted in this test, so applying the
        # gate vs not should yield the same numbers — but the code
        # path being exercised is what we care about.
        result_gated = svc.close_epoch(apply_gate=True)
        svc.open_epoch()
        svc.submit_observation(_obs("intel2", charter=(0.0, 0, 0.9, 0), embed_idx=2), agent_id="A")
        result_raw = svc.close_epoch(apply_gate=False)
        # Both should carry positive score-movement diagnostics (gate
        # isn't suppressing anything in this clean scenario); neither
        # mints — no tool activity happened.
        assert result_gated["score_agent_mint"].get("A", 0) > 0
        assert result_raw["score_agent_mint"].get("A", 0) > 0
        assert result_gated["agent_mint"] == {}
        assert result_raw["agent_mint"] == {}
    finally:
        svc.shutdown()


def test_epoch_history_records_closed_epochs(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_epoch_g", data_root=tmp_path)
    try:
        for i in range(3):
            svc.open_epoch(f"epoch_{i}")
            svc.submit_observation(_obs(f"e{i}", charter=(0.0, 0, 0.5, 0), embed_idx=i), agent_id="A")
            svc.close_epoch()
        history = svc.epoch_history
        assert len(history) == 3
        assert [h["epoch_id"] for h in history] == ["epoch_0", "epoch_1", "epoch_2"]
        for h in history:
            assert h["closed_at"] >= h["started_at"]
    finally:
        svc.shutdown()


def test_checkpoint_records_intermediate_snapshot(tmp_path: Path):
    svc = WorldService(rpb_address="rpb_epoch_h", data_root=tmp_path)
    try:
        svc.open_epoch()
        svc.submit_observation(_obs("a", charter=(0.0, 0, 0.5, 0)), agent_id="A")
        cp = svc.checkpoint_epoch()
        assert cp is not None
        assert cp["checkpoint_index"] == 0
        svc.submit_observation(_obs("b", charter=(0.0, 0, 0.5, 0), embed_idx=1), agent_id="A")
        cp2 = svc.checkpoint_epoch()
        assert cp2["checkpoint_index"] == 1
        svc.close_epoch()
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Tool rail (one earning rail: local projection matches federated close)
# ---------------------------------------------------------------------------


def test_local_close_mints_through_tool_rail_only(tmp_path: Path):
    """Registration + attested receipts mint to the tool author at the
    local close; concurrent prose activity earns nothing."""
    svc = WorldService(
        rpb_address="rpb_epoch_tools", data_root=tmp_path,
        embedding_dim=_TOOL_EMBED_DIM,
    )
    try:
        svc.open_epoch("e_tools")
        svc.submit_events(
            [_tool_registration_event(1)], equilibrate_after=False,
        )
        svc.submit_events(
            [_tool_vet_event(2, "vetter-1"),
             _tool_vet_event(3, "vetter-2"),
             _tool_receipt_event(4, "caller-1"),
             _tool_receipt_event(5, "caller-2")],
            equilibrate_after=False,
        )
        svc.submit_observation(
            _obs("prose", charter=(0.0, 0, 0.7, 0)), agent_id="A",
        )
        result = svc.close_epoch()
        assert result["agent_mint"].get(_TOOL_AUTHOR, 0) > 0, result
        assert "A" not in result["agent_mint"]
        assert result["total_mint"] == pytest.approx(
            sum(result["agent_mint"].values()))
        entry = result["tool_mint"][_TOOL_DIGEST]
        assert entry["author"] == _TOOL_AUTHOR
        assert entry["attesters"] == 2
        # Earnings surface (agent projection) sees the tool mint.
        proj = svc.read_agent_projection(_TOOL_AUTHOR, last_n_epochs=5)
        assert proj["total_mint_projection"] > 0
        proj_a = svc.read_agent_projection("A", last_n_epochs=5)
        assert proj_a["total_mint_projection"] == 0.0
    finally:
        svc.shutdown()


def test_tool_registration_carries_across_local_epochs(tmp_path: Path):
    """A tool registered + greenlit in an earlier epoch still
    attributes mint when receipts land in a later epoch (registration
    and vetting carry-over maps).

    NOTE: in-process only — the world checkpoint doesn't persist
    ``artifact_digest`` on nodes yet, so cross-RESTART standing is a
    known persistence gap (the carry-over maps themselves do survive
    restart via local_tool_registrations.json / local_tool_vetting.json).
    """
    rpb = "rpb_epoch_tools_carry"
    svc = WorldService(
        rpb_address=rpb, data_root=tmp_path,
        embedding_dim=_TOOL_EMBED_DIM,
    )
    try:
        svc.open_epoch("e_reg")
        svc.submit_events(
            [_tool_registration_event(1),
             _tool_vet_event(2, "vetter-1"),
             _tool_vet_event(3, "vetter-2")],
            equilibrate_after=False,
        )
        r1 = svc.close_epoch()
        # Registration + greenlight alone: no usage, no mint.
        assert r1["agent_mint"] == {}

        svc.open_epoch("e_use")
        svc.submit_events(
            [_tool_receipt_event(4, "caller-1")], equilibrate_after=False,
        )
        r2 = svc.close_epoch()
        assert r2["agent_mint"].get(_TOOL_AUTHOR, 0) > 0, r2
        assert r2["tool_mint"][_TOOL_DIGEST]["author"] == _TOOL_AUTHOR
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_scheduler_force_tick_triggers_close_open(tmp_path: Path):
    from nodes.common.epoch_scheduler import (
        EpochScheduler,
        EpochSchedulerConfig,
    )
    svc = WorldService(rpb_address="rpb_epoch_sched_a", data_root=tmp_path)
    received: List[Dict[str, Any]] = []
    try:
        sched = EpochScheduler(
            world_service=svc,
            config=EpochSchedulerConfig(interval_seconds=3600.0),
            on_close=received.append,
        )
        # Scheduler opened the first epoch on construction.
        assert svc.current_epoch_id is not None
        first_epoch_id = svc.current_epoch_id
        svc.submit_observation(_obs("a", charter=(0.0, 0, 0.6, 0)), agent_id="X")
        result = sched.force_tick()
        # The handler got the close result (novelty diagnostic — prose
        # activity doesn't mint on the tool-only rail).
        assert received and received[0]["agent_novelty"].get("X", 0) > 0
        # And a new epoch is open with a different id.
        assert svc.current_epoch_id is not None
        assert svc.current_epoch_id != first_epoch_id
        assert sched.closed_count == 1
    finally:
        svc.shutdown()


def test_scheduler_maybe_tick_respects_interval(tmp_path: Path):
    from nodes.common.epoch_scheduler import (
        EpochScheduler,
        EpochSchedulerConfig,
    )
    svc = WorldService(rpb_address="rpb_epoch_sched_b", data_root=tmp_path)
    closes: List[Dict[str, Any]] = []
    try:
        sched = EpochScheduler(
            world_service=svc,
            config=EpochSchedulerConfig(interval_seconds=10.0),
            on_close=closes.append,
        )
        # First tick: too soon — no-op.
        assert sched.maybe_tick() is None
        assert sched.closed_count == 0
        # Pretend the interval has passed.
        sched._last_open_at = time.time() - 11.0
        result = sched.maybe_tick()
        assert result is not None
        assert sched.closed_count == 1
    finally:
        svc.shutdown()


def test_scheduler_handler_failure_is_isolated(tmp_path: Path):
    """A buggy handler must not crash the scheduler — its job is to
    drive the world's epoch boundaries regardless of what consumers
    do with the result."""
    from nodes.common.epoch_scheduler import (
        EpochScheduler,
        EpochSchedulerConfig,
    )
    svc = WorldService(rpb_address="rpb_epoch_sched_c", data_root=tmp_path)

    def _broken(_):
        raise RuntimeError("handler explodes")

    try:
        sched = EpochScheduler(
            world_service=svc,
            config=EpochSchedulerConfig(interval_seconds=3600.0),
            on_close=_broken,
        )
        # Should not raise.
        sched.force_tick()
        # Scheduler should have opened a new epoch despite the handler error.
        assert svc.current_epoch_id is not None
    finally:
        svc.shutdown()
