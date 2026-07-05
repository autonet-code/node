"""Tool usage receipts — ToolUsed events + deterministic aggregation.

Covers the event roundtrip, replay neutrality (receipts never touch the
claim graph), and tool_usage_from_events determinism under input
shuffling. Design: docs/tool_substrate.md.
"""

from __future__ import annotations

import random

from nodes.common.world_model_substrate.adapter import build_charter_world
from nodes.common.world_model_substrate.aggregate import apply_events
from nodes.common.world_model_substrate.events import (
    SubClaimSprouted,
    ToolUsed,
    event_from_dict,
)
from nodes.common.world_model_substrate.tool_usage import tool_usage_from_events


def _receipt(caller, digest, seq, *, ok=True, fee=0.0, author="toolsmith"):
    return ToolUsed(
        seq=seq,
        author_agent=caller,
        manifest_digest=digest,
        tool_author=author,
        receipt_digest=f"r{seq:02d}" * 8,
        ok=ok,
        fee_atn=fee,
    ).to_dict()


def test_event_roundtrip():
    ev = _receipt("agent-a", "d" * 64, 3, fee=0.25)
    restored = event_from_dict(ev)
    assert isinstance(restored, ToolUsed)
    assert restored.to_dict() == ev


def test_replay_ignores_receipts():
    """tool_used events must not alter the claim graph in any way."""
    world = build_charter_world()
    before = {t.id: t.tree.root_node.id for t in world.tendencies.values()}
    n_before = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())

    apply_events(world, [
        _receipt("agent-a", "d" * 64, 1),
        _receipt("agent-b", "e" * 64, 2, ok=False),
    ], equilibrate_after=False)

    n_after = sum(len(t.tree.all_nodes()) for t in world.tendencies.values())
    assert n_after == n_before
    assert {t.id: t.tree.root_node.id for t in world.tendencies.values()} == before


def test_usage_aggregation():
    d1, d2 = "a" * 64, "b" * 64
    events = [
        _receipt("caller-1", d1, 1, fee=0.5),
        _receipt("caller-2", d1, 1, fee=0.5),
        _receipt("caller-1", d1, 2, fee=0.5, ok=False),   # failure: counted, not ok
        _receipt("caller-1", d2, 3, author="other-author"),
        {"kind": "observation_added", "obs_id": "x"},      # non-receipt ignored
    ]
    usage = tool_usage_from_events(events)

    assert list(usage.keys()) == [d1, d2]  # canonical (sorted) order
    assert usage[d1]["count"] == 3
    assert usage[d1]["ok_count"] == 2
    assert usage[d1]["fee_total"] == 1.5
    assert usage[d1]["tool_author"] == "toolsmith"
    assert usage[d1]["callers"] == {"caller-1": 2, "caller-2": 1}
    assert usage[d2]["tool_author"] == "other-author"


def test_aggregation_deterministic_under_shuffle():
    digests = ["a" * 64, "b" * 64, "c" * 64]
    events = [
        _receipt(f"caller-{i % 4}", digests[i % 3], i, fee=0.1 * (i % 5))
        for i in range(40)
    ]
    baseline = tool_usage_from_events(list(events))
    rng = random.Random(7)
    for _ in range(5):
        shuffled = list(events)
        rng.shuffle(shuffled)
        assert tool_usage_from_events(shuffled) == baseline


def test_vet_events_aggregate_separately_from_usage():
    """Vets (third flavor) accumulate in vets_by_caller and never touch
    usage counts; a negative vet is debate material, not a greenlight
    input (spec: Vetting section)."""
    d = "a" * 64
    vet_yes = ToolUsed(seq=1, author_agent="validator-1", manifest_digest=d,
                       receipt_digest="v1" * 16, ok=True, vet=True).to_dict()
    vet_yes["sender"] = "aa11"
    vet_no = ToolUsed(seq=2, author_agent="validator-2", manifest_digest=d,
                      receipt_digest="v2" * 16, ok=False, vet=True).to_dict()
    events = [vet_yes, vet_no, _receipt("caller-1", d, 3)]
    usage = tool_usage_from_events(events)

    assert usage[d]["count"] == 1                     # the real receipt only
    assert usage[d]["callers"] == {"caller-1": 1}
    assert usage[d]["vets_by_caller"] == {"validator-1": 1}
    assert usage[d]["vet_senders"] == {"validator-1": ["aa11"]}


def test_vet_field_serializes_only_when_true():
    """Back-compat: pre-vetting event logs must hash identically."""
    plain = ToolUsed(seq=1, author_agent="c", manifest_digest="b" * 64,
                     receipt_digest="r" * 32).to_dict()
    assert "vet" not in plain
    vet = ToolUsed(seq=1, author_agent="c", manifest_digest="b" * 64,
                   receipt_digest="r" * 32, vet=True).to_dict()
    assert vet["vet"] is True
    restored = event_from_dict(vet)
    assert isinstance(restored, ToolUsed) and restored.vet


def test_evidence_field_serializes_only_when_present():
    """Evidence-replay CON (docs/tool_substrate.md — Evidence): the
    optional evidence dict rides a CON sprout serialize-only-when-present,
    so pre-evidence sprout logs hash byte-identically."""
    plain = SubClaimSprouted(
        seq=2, author_agent="critic", tendency_id="correctness",
        parent_id="tm_abc", node_id="con_xyz", position="con",
        content="disputes the tool", author_post=True).to_dict()
    assert "evidence" not in plain

    ev = SubClaimSprouted(
        seq=2, author_agent="critic", tendency_id="correctness",
        parent_id="tm_abc", node_id="con_xyz", position="con",
        content="disputes the tool", author_post=True,
        evidence={"args_json": "{\"x\": \"\"}",
                  "expected_error": "x is required",
                  "actual_digest": "de" * 32}).to_dict()
    assert ev["evidence"]["expected_error"] == "x is required"

    # Round-trip: event_from_dict reconstructs the evidence-bearing sprout.
    restored = event_from_dict(ev)
    assert isinstance(restored, SubClaimSprouted)
    assert restored.evidence["args_json"] == "{\"x\": \"\"}"
    assert restored.to_dict() == ev

    # An empty evidence dict serializes AWAY (back-compat hashing).
    empty = SubClaimSprouted(
        seq=2, author_agent="critic", tendency_id="correctness",
        parent_id="tm_abc", node_id="con_xyz", position="con",
        content="disputes the tool", evidence={}).to_dict()
    assert "evidence" not in empty
