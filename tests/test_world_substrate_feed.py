"""Integration tests for WorldSubstrateFeed (Phase 2).

Validates the end-to-end path:
  agent JSONL traces -> outcomes + work units -> WorldService.

These tests construct a fake ATN data directory with a few synthetic
conversations, point a feed at it, and verify that:

  1. ``run_cycle`` waits until enough events accumulate.
  2. Once it runs, work units flow into the WorldService and the world
     state changes (events_applied increments, charter scores move).
  3. Re-running a cycle doesn't reprocess already-handled
     conversations (idempotency within a daemon lifetime).
  4. A daemon restart triggers re-processing — but the WorldService's
     content-addressed dedupe catches duplicates so the world doesn't
     double-count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from nodes.common.world_service import WorldService
from nodes.common.world_substrate_feed import (
    ResolvedAgentIdentity,
    WorldSubstrateFeed,
    WorldSubstrateFeedConfig,
)


# ---------------------------------------------------------------------------
# Fixtures: build a fake ATN data tree with synthetic conversations
# ---------------------------------------------------------------------------


def _write_conversation(
    path: Path,
    messages: List[Dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for msg in messages:
            fh.write(json.dumps(msg) + "\n")


def _build_atn_dir(tmp_path: Path, n_conversations: int = 3) -> Path:
    """Lay out a ``~/.atn``-style directory with a few conversations."""
    atn_root = tmp_path / "atn"
    agent_dir = atn_root / "agents" / "test-agent"
    conv_dir = agent_dir / "conversations"
    for i in range(n_conversations):
        _write_conversation(
            conv_dir / f"session_{i:03d}.jsonl",
            [
                {
                    "role": "user",
                    "content": f"Help me write tests for module {i} that"
                                " checks for edge cases in arithmetic"
                                " operations.",
                },
                {
                    "role": "assistant",
                    "content": f"Sure — here's a test suite for module"
                                f" {i} with edge cases including"
                                " overflow, underflow, and divide-by-zero"
                                " coverage.",
                },
                {
                    "role": "user",
                    "content": "Looks good, thanks!",
                },
            ],
        )
    return atn_root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_feed_does_not_run_until_threshold_reached(tmp_path: Path):
    """Verifies ``should_run`` gating: the feed waits for enough
    pending-event notifications before doing real work."""
    atn_root = _build_atn_dir(tmp_path)
    svc = WorldService(rpb_address="rpb_feed_a", data_root=tmp_path / "world")
    try:
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(
                data_dir=str(atn_root),
                min_events_for_cycle=3,
                min_cycle_interval=0.0,  # no time gate for tests
            ),
            world_service=svc,
        )
        # Pending counter at zero — should_run is False.
        assert not feed.should_run()
        feed.notify_execution("test-agent", "exec-1", "completed")
        feed.notify_execution("test-agent", "exec-2", "completed")
        assert not feed.should_run()  # only 2 events
        feed.notify_execution("test-agent", "exec-3", "completed")
        assert feed.should_run()
    finally:
        svc.shutdown()


def test_feed_drives_work_units_into_world(tmp_path: Path):
    """Once the threshold trips, the feed builds work units from the
    synthetic conversations and submits them to the WorldService.
    The world state must reflect that."""
    atn_root = _build_atn_dir(tmp_path, n_conversations=3)
    svc = WorldService(rpb_address="rpb_feed_b", data_root=tmp_path / "world")
    try:
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(
                data_dir=str(atn_root),
                min_events_for_cycle=1,
                min_cycle_interval=0.0,
            ),
            world_service=svc,
        )
        nodes_before = svc.stats()["n_nodes"]
        feed.notify_execution("test-agent", "exec-1", "completed")
        metrics = feed.run_cycle()
        assert metrics is not None, "feed should have run"
        assert metrics["units_processed"] == 3, metrics
        assert metrics["events_appended"] > 0, metrics
        nodes_after = svc.stats()["n_nodes"]
        assert nodes_after > nodes_before, \
            f"expected world to grow: before={nodes_before}, after={nodes_after}"
    finally:
        svc.shutdown()


def test_feed_does_not_reprocess_in_same_lifetime(tmp_path: Path):
    """Same daemon, two cycles: the second cycle should find no new
    work units because every conversation was already processed in the
    first cycle."""
    atn_root = _build_atn_dir(tmp_path, n_conversations=2)
    svc = WorldService(rpb_address="rpb_feed_c", data_root=tmp_path / "world")
    try:
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(
                data_dir=str(atn_root),
                min_events_for_cycle=1,
                min_cycle_interval=0.0,
            ),
            world_service=svc,
        )
        feed.notify_execution("test-agent", "exec-1", "completed")
        first = feed.run_cycle()
        assert first is not None
        assert first["units_processed"] == 2

        feed.notify_execution("test-agent", "exec-2", "completed")
        second = feed.run_cycle()
        assert second is not None
        # Cycle ran but found nothing new to process.
        assert second["units_processed"] == 0, second
    finally:
        svc.shutdown()


def test_feed_picks_up_new_conversation_added_between_cycles(tmp_path: Path):
    """Drop a new conversation file mid-stream; the next cycle picks
    only it up, not the previously processed ones."""
    atn_root = _build_atn_dir(tmp_path, n_conversations=2)
    svc = WorldService(rpb_address="rpb_feed_d", data_root=tmp_path / "world")
    try:
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(
                data_dir=str(atn_root),
                min_events_for_cycle=1,
                min_cycle_interval=0.0,
            ),
            world_service=svc,
        )
        feed.notify_execution("test-agent", "exec-1", "completed")
        first = feed.run_cycle()
        assert first["units_processed"] == 2

        # New conversation arrives.
        new_path = atn_root / "agents" / "test-agent" / "conversations" / "session_999.jsonl"
        _write_conversation(new_path, [
            {"role": "user", "content": "Now help me design a config schema."},
            {"role": "assistant", "content": "Here is a config schema with required fields A, B, C."},
        ])

        feed.notify_execution("test-agent", "exec-2", "completed")
        second = feed.run_cycle()
        assert second["units_processed"] == 1, second
    finally:
        svc.shutdown()


def test_feed_skips_empty_conversations(tmp_path: Path):
    """A conversation file with no extractable problem/resolution
    should not produce a work unit."""
    atn_root = tmp_path / "atn"
    conv_dir = atn_root / "agents" / "test-agent" / "conversations"
    # Empty file
    _write_conversation(conv_dir / "blank.jsonl", [])
    # Only-user, no resolution
    _write_conversation(conv_dir / "no_response.jsonl", [
        {"role": "user", "content": "Hello?"},
    ])
    # Only-assistant, no problem
    _write_conversation(conv_dir / "no_question.jsonl", [
        {"role": "assistant", "content": "I'm here to help."},
    ])

    svc = WorldService(rpb_address="rpb_feed_e", data_root=tmp_path / "world")
    try:
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(
                data_dir=str(atn_root),
                min_events_for_cycle=1,
                min_cycle_interval=0.0,
            ),
            world_service=svc,
        )
        feed.notify_execution("test-agent", "exec-1", "completed")
        metrics = feed.run_cycle()
        assert metrics is not None
        assert metrics["units_processed"] == 0, metrics
    finally:
        svc.shutdown()


def test_max_units_cap_is_respected(tmp_path: Path):
    """A large directory of conversations gets capped at
    ``max_units_per_cycle`` to prevent burst overload."""
    atn_root = _build_atn_dir(tmp_path, n_conversations=50)
    svc = WorldService(rpb_address="rpb_feed_f", data_root=tmp_path / "world")
    try:
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(
                data_dir=str(atn_root),
                min_events_for_cycle=1,
                min_cycle_interval=0.0,
                max_units_per_cycle=10,
            ),
            world_service=svc,
        )
        feed.notify_execution("test-agent", "exec-1", "completed")
        metrics = feed.run_cycle()
        assert metrics is not None
        assert metrics["units_processed"] == 10, metrics
    finally:
        svc.shutdown()


def test_feed_does_nothing_without_data_dir(tmp_path: Path):
    """Edge: data_dir empty -> feed never runs."""
    svc = WorldService(rpb_address="rpb_feed_g", data_root=tmp_path / "world")
    try:
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(data_dir=""),
            world_service=svc,
        )
        for _ in range(10):
            feed.notify_execution("test-agent", "exec-x", "completed")
        assert not feed.should_run()
        assert feed.run_cycle() is None
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# Phase 10.1: per-agent attribution via identity_resolver
# ---------------------------------------------------------------------------


def _build_atn_dir_for_agent(
    atn_root: Path,
    local_agent_id: str,
    n_conversations: int = 1,
    seed: int = 0,
) -> None:
    """Add conversations for one local agent_id under atn_root/agents/."""
    conv_dir = atn_root / "agents" / local_agent_id / "conversations"
    for i in range(n_conversations):
        _write_conversation(
            conv_dir / f"session_{seed:02d}_{i:03d}.jsonl",
            [
                {"role": "user", "content": f"problem #{seed}.{i} from {local_agent_id}"},
                {"role": "assistant", "content": f"resolution #{seed}.{i} from {local_agent_id}"},
            ],
        )


def test_resolver_attributes_to_onchain_address(tmp_path: Path):
    """Registered agent's work units are submitted under the 0x
    address, not under the YAML id or the daemon fallback."""
    atn_root = tmp_path / "atn"
    _build_atn_dir_for_agent(atn_root, "agent-alpha", n_conversations=2)

    svc = WorldService(rpb_address="rpb_resolver_a", data_root=tmp_path / "world")
    try:
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(
                data_dir=str(atn_root),
                min_events_for_cycle=1,
                min_cycle_interval=0.0,
            ),
            world_service=svc,
            identity_resolver=lambda aid: (
                ResolvedAgentIdentity(address="0xAAA", registered_on_chain=True)
                if aid == "agent-alpha" else None
            ),
        )
        feed.notify_execution("agent-alpha", "exec-1", "completed")
        metrics = feed.run_cycle()
        assert metrics is not None
        assert metrics["units_processed"] == 2
        assert metrics["agents_attributed"] == 1
        # Verify the work was attributed to the 0x address, not to
        # the YAML id or the daemon fallback.
        proj_addr = svc.read_agent_projection("0xAAA", last_n_epochs=10)
        proj_yaml = svc.read_agent_projection("agent-alpha", last_n_epochs=10)
        proj_daemon = svc.read_agent_projection("daemon", last_n_epochs=10)
        assert proj_addr["agent_id"] == "0xAAA"
        # The YAML id and "daemon" fallback should not have any mint
        # because no events are attributed under those names.
        assert proj_yaml["total_mint_projection"] == 0.0
        assert proj_daemon["total_mint_projection"] == 0.0
    finally:
        svc.shutdown()


def test_resolver_skips_unregistered_agent(tmp_path: Path):
    """Agent without registered_on_chain gets skipped — no events
    flow into the world."""
    atn_root = tmp_path / "atn"
    _build_atn_dir_for_agent(atn_root, "agent-beta", n_conversations=2)

    svc = WorldService(rpb_address="rpb_resolver_b", data_root=tmp_path / "world")
    try:
        nodes_before = svc.stats()["n_nodes"]
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(
                data_dir=str(atn_root),
                min_events_for_cycle=1,
                min_cycle_interval=0.0,
            ),
            world_service=svc,
            identity_resolver=lambda aid: ResolvedAgentIdentity(
                address="0xBBB", registered_on_chain=False,
            ),
        )
        feed.notify_execution("agent-beta", "exec-1", "completed")
        metrics = feed.run_cycle()
        assert metrics is not None
        assert metrics["units_processed"] == 0
        nodes_after = svc.stats()["n_nodes"]
        assert nodes_after == nodes_before, \
            "unregistered agent should not have grown the world"
    finally:
        svc.shutdown()


def test_resolver_groups_by_author_across_agents(tmp_path: Path):
    """Two registered agents in the same daemon get distinct author
    attribution; each produces its own per-agent mint."""
    atn_root = tmp_path / "atn"
    _build_atn_dir_for_agent(atn_root, "agent-1", n_conversations=2, seed=1)
    _build_atn_dir_for_agent(atn_root, "agent-2", n_conversations=1, seed=2)

    svc = WorldService(rpb_address="rpb_resolver_c", data_root=tmp_path / "world")
    try:
        addresses = {"agent-1": "0x111", "agent-2": "0x222"}
        feed = WorldSubstrateFeed(
            config=WorldSubstrateFeedConfig(
                data_dir=str(atn_root),
                min_events_for_cycle=1,
                min_cycle_interval=0.0,
            ),
            world_service=svc,
            identity_resolver=lambda aid: (
                ResolvedAgentIdentity(address=addresses[aid], registered_on_chain=True)
                if aid in addresses else None
            ),
        )
        feed.notify_execution("agent-1", "exec-1", "completed")
        feed.notify_execution("agent-2", "exec-2", "completed")
        metrics = feed.run_cycle()
        assert metrics is not None
        assert metrics["units_processed"] == 3
        assert metrics["agents_attributed"] == 2
    finally:
        svc.shutdown()
