"""v3 per-axis review scores (docs/tool_substrate.md, Decision 2026-07-08).

Covers the additive ToolUsed.axes field: legacy events must serialize
BYTE-IDENTICALLY (gossip batch hashes are load-bearing), axes serialize
sorted+clamped when present, and tool_usage_from_events aggregates
per-caller per-axis sums deterministically.
"""

from __future__ import annotations

import json

from nodes.common.world_model_substrate.events import ToolUsed, event_from_dict
from nodes.common.world_model_substrate.tool_usage import tool_usage_from_events

DIGEST = "ab" * 32


def _receipt(**over):
    kwargs = dict(
        seq=1,
        author_agent="caller-1",
        manifest_digest=DIGEST,
        tool_author="author-1",
        receipt_digest="r1" * 8,
        ok=True,
        fee_atn=0.0,
        attested=True,
        score=0.8,
    )
    kwargs.update(over)
    return ToolUsed(**kwargs)


class TestWireBackCompat:
    # Pinned pre-v3 serialization of an attested receipt. If this
    # fixture ever fails, gossip batch hashes for legacy event logs
    # have forked — that is a consensus break, not a test to update.
    LEGACY = {
        "kind": "tool_used",
        "seq": 1,
        "author_agent": "caller-1",
        "manifest_digest": DIGEST,
        "tool_author": "author-1",
        "receipt_digest": "r1" * 8,
        "ok": True,
        "fee_atn": 0.0,
        "attested": True,
        "score": 0.8,
    }

    def test_axes_free_event_serializes_byte_identically(self):
        d = _receipt().to_dict()
        assert d == self.LEGACY
        assert json.dumps(d, sort_keys=True) == json.dumps(
            self.LEGACY, sort_keys=True)

    def test_mechanical_receipt_unchanged(self):
        d = _receipt(attested=False, score=0.0).to_dict()
        assert "axes" not in d and "attested" not in d and "score" not in d

    def test_axes_serialize_sorted_and_clamped(self):
        d = _receipt(axes={"simplicity": 2.5, "correctness": -3.0,
                           "evolution": 0.25}).to_dict()
        assert list(d["axes"].keys()) == [
            "correctness", "evolution", "simplicity"]
        assert d["axes"]["simplicity"] == 1.0
        assert d["axes"]["correctness"] == -1.0
        assert d["axes"]["evolution"] == 0.25

    def test_from_dict_round_trip(self):
        original = _receipt(axes={"correctness": 0.9})
        restored = event_from_dict(original.to_dict())
        assert isinstance(restored, ToolUsed)
        assert restored.axes == {"correctness": 0.9}
        # Legacy dict without axes → empty dict, not an error.
        legacy = event_from_dict(dict(self.LEGACY))
        assert legacy.axes == {}


class TestAxisAggregation:
    def test_per_caller_per_axis_sums(self):
        events = [
            _receipt(seq=1, author_agent="a",
                     axes={"correctness": 0.8, "simplicity": -0.2}).to_dict(),
            _receipt(seq=2, author_agent="a", receipt_digest="r2" * 8,
                     axes={"correctness": 0.6}).to_dict(),
            _receipt(seq=3, author_agent="b", receipt_digest="r3" * 8,
                     axes={"correctness": -1.0}).to_dict(),
            # Axis-less attestation still counts as usage, no axis rows.
            _receipt(seq=4, author_agent="c", receipt_digest="r4" * 8).to_dict(),
        ]
        usage = tool_usage_from_events(events)
        entry = usage[DIGEST]
        reviews = entry["axis_reviews_by_caller"]
        assert reviews["a"]["correctness"] == {"sum": 1.4, "n": 2}
        assert reviews["a"]["simplicity"] == {"sum": -0.2, "n": 1}
        assert reviews["b"]["correctness"] == {"sum": -1.0, "n": 1}
        assert "c" not in reviews
        # Usage counting unaffected by axes presence.
        assert entry["attested_ok_by_caller"] == {"a": 2, "b": 1, "c": 1}

    def test_vets_and_failures_carry_no_axes(self):
        events = [
            _receipt(seq=1, author_agent="v", vet=True,
                     axes={"correctness": 1.0}).to_dict(),
            _receipt(seq=2, author_agent="a", ok=False,
                     receipt_digest="r2" * 8,
                     axes={"correctness": -1.0}).to_dict(),
        ]
        entry = tool_usage_from_events(events)[DIGEST]
        # Vet excluded entirely; failed receipt is not attested-ok.
        assert entry["axis_reviews_by_caller"] == {}

    def test_deterministic_across_input_order(self):
        events = [
            _receipt(seq=i, author_agent=f"agent-{i % 3}",
                     receipt_digest=f"r{i:02d}" * 8,
                     axes={"correctness": (-1) ** i * 0.5,
                           "evolution": 0.1 * (i % 5)}).to_dict()
            for i in range(1, 10)
        ]
        forward = tool_usage_from_events(list(events))
        backward = tool_usage_from_events(list(reversed(events)))
        assert json.dumps(forward, sort_keys=True) == json.dumps(
            backward, sort_keys=True)
