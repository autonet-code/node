"""Tool positions (docs/multi_jurisdiction.md decision 10, replacing the
v3 gossip-review drift of Decision 2026-07-08).

A tool's charter head is its ToolBond rating pinned to the previous anchor:
per axis the REP-weighted mean (sum / mass / 100), mass = REP mass / REP
supply. Gossiped review axes no longer move it. Covers: entry, ratings in,
gossip axes ignored, and bit-identical replay through the close.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import (
    compute_tool_mint,
    federated_epoch_close,
)
from nodes.common.library.charter import DEFAULT_CHARTER

N_DIMS = len(DEFAULT_CHARTER.ids)
DIGEST = "ab" * 32
AUTHOR = "toolsmith"


def _registration_event(seq: int = 1, *, digest: str = DIGEST,
                        author: str = AUTHOR) -> Dict[str, Any]:
    return {
        "kind": "tool_registered",
        "seq": seq,
        "author_agent": author,
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": "pinned", "author": author},
    }


def _receipt(seq: int, caller: str, *, digest: str = DIGEST,
             axes: Dict[str, float] | None = None) -> Dict[str, Any]:
    ev = {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": caller,
        "manifest_digest": digest,
        "tool_author": AUTHOR,
        "receipt_digest": f"r{seq:02d}" * 8,
        "ok": True,
        "fee_atn": 0.0,
        "attested": True,
        "score": 0.8,
    }
    if axes:
        ev["axes"] = dict(axes)
    return ev


REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}


SUPPLY = 1000


def _snap(sum_: int, mass: int, axis: str = "correctness") -> Dict[str, Any]:
    return {"supply": SUPPLY, "tools": {DIGEST: {
        "axes": {axis: {"sum": sum_, "mass": mass}}, "harm": {"sum": 0, "mass": 0}}}}


class TestPositionsFromRatings:
    def test_new_tool_enters_at_zero(self):
        out = compute_tool_mint([], registrations=REGS)
        pos = out["positions_next"][DIGEST]
        assert pos["head"] == [0.0] * N_DIMS
        assert pos["mass"] == [0.0] * N_DIMS

    def test_rating_sets_scored_axis_only(self):
        out = compute_tool_mint([], registrations=REGS,
                                review_snapshot=_snap(80 * 300, 300))
        pos = out["positions_next"][DIGEST]
        # correctness is axis_index 4 (adapter CHARTER order).
        assert pos["head"][4] == pytest.approx(0.8)
        assert pos["mass"][4] == pytest.approx(0.3)
        for i in range(N_DIMS):
            if i != 4:
                assert pos["head"][i] == 0.0 and pos["mass"][i] == 0.0

    def test_gossip_review_axes_do_not_drift(self):
        out = compute_tool_mint(
            [_receipt(1, "caller-1", axes={"correctness": 0.8})],
            registrations=REGS)
        assert out["positions_next"][DIGEST]["head"] == [0.0] * N_DIMS

    def test_carried_head_is_replaced_by_the_rating(self):
        out = compute_tool_mint([_receipt(1, "caller-1")],
                                registrations=REGS,
                                review_snapshot=_snap(-40 * 10, 10))
        pos = out["positions_next"][DIGEST]
        assert pos["head"][4] == pytest.approx(-0.4)
        assert pos["head"][0] == 0.0


class TestCloseIntegration:
    def _batches(self, events, kp):
        chain, prev = [], b""
        for i, ev in enumerate(events, start=1):
            b = EventBatch(
                rpb_address="rpb_drift", sender_pubkey=kp.public_key,
                batch_seq=i, events=[ev], prev_batch_hash=prev,
                timestamp=1_700_000_000.0 + i)
            chain.append(b)
            prev = b.content_hash()
        return chain

    def _standard(self):
        kp_a, kp_c = Keypair.generate(), Keypair.generate()
        batches = self._batches([_registration_event(1)], kp_a)
        batches += self._batches(
            [_receipt(1, "caller-1", axes={"correctness": 0.9,
                                           "simplicity": 0.5})], kp_c)
        return batches

    def test_close_returns_and_replays_positions(self):
        batches = self._standard()
        snap = _snap(90 * 50, 50)
        r1 = federated_epoch_close(canonical_order(list(batches)),
                                   review_snapshot=snap)
        r2 = federated_epoch_close(canonical_order(list(batches)),
                                   review_snapshot=snap)
        assert DIGEST in r1["tool_positions"]
        assert r1["tool_positions"][DIGEST]["head"][4] == pytest.approx(0.9)
        assert json.dumps(r1["tool_positions"]) == json.dumps(
            r2["tool_positions"])

    def test_next_close_reads_the_next_snapshot(self):
        batches = self._standard()
        r1 = federated_epoch_close(canonical_order(list(batches)),
                                   review_snapshot=_snap(90 * 50, 50))
        kp_c = Keypair.generate()
        b2 = self._batches([_receipt(1, "caller-2")], kp_c)
        # A later round finalized: the rating on chain moved.
        r2 = federated_epoch_close(
            canonical_order(b2),
            tool_registrations=r1["tool_registrations"],
            review_snapshot=_snap(90 * 50 - 90 * 50, 100),
        )
        assert r2["tool_positions"][DIGEST]["head"][4] == 0.0
        assert r2["tool_positions"][DIGEST]["mass"][4] == pytest.approx(0.1)
