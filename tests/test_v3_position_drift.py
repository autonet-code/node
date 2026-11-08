"""Tool positions (docs/multi_jurisdiction.md decision 10, replacing the
v3 gossip-review drift of Decision 2026-07-08).

A tool's charter head is its ToolBond rating pinned to the previous anchor:
per axis the REP-weighted mean (sum / mass / 100), mass = REP mass / REP
supply. Gossiped review axes no longer move it. Covers: entry, ratings in,
gossip axes ignored, world application (claim anchors + observation coords),
and bit-identical replay through the close.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import (
    apply_tool_positions,
    compute_tool_mint,
    federated_epoch_close,
)
from nodes.common.world_model_substrate.adapter import (
    N_DIMS,
    build_charter_world,
)
from nodes.common.world_model_substrate.aggregate import apply_events

EMBED_DIM = 64
DIGEST = "ab" * 32
AUTHOR = "toolsmith"


def _coords(axis: int = 4, mag: float = 0.8) -> List[float]:
    out = [0.0] * (N_DIMS + EMBED_DIM)
    out[axis] = mag
    out[N_DIMS + 3] = 0.5
    return out


def _registration_event(seq: int = 1, *, digest: str = DIGEST,
                        author: str = AUTHOR) -> Dict[str, Any]:
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": author,
        "tendency_id": "correctness",
        "parent_id": "solver_root",
        "node_id": f"tm_{digest[:12]}",
        "position": "pro",
        "coords": _coords(),
        "polarity_axis": _coords(),
        "content": f"tool {digest[:8]}: does a thing",
        "author_post": True,
        "artifact_digest": digest,
        "observation_id": "tm_" + digest[:16],
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


def _world_with_registration():
    world = build_charter_world(embedding_dim=EMBED_DIM)
    apply_events(world, [_registration_event(1)], equilibrate_after=False)
    return world


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


class TestWorldApplication:
    def test_apply_updates_claim_anchor_and_observation(self):
        world = _world_with_registration()
        # Replay worlds carry no Observation entries (only sprout claims);
        # the LIVE path registers one. Insert it so both coordinate
        # stores are exercised, as they are on a live daemon.
        from world_model.generalized.observation import Observation
        for tendency in world.tendencies.values():
            for node in tendency.tree.all_nodes():
                if getattr(node, "artifact_digest", "") == DIGEST \
                        and getattr(node, "observation_id", None):
                    world.observations[node.observation_id] = Observation(
                        id=node.observation_id,
                        coords=tuple(_coords()),
                        label="tool claim",
                    )
        head = [0.0, 0.0, 0.0, 0.0, 0.55, -0.2]
        n = apply_tool_positions(
            world, {DIGEST: {"head": head, "mass": [2.0] * N_DIMS}})
        assert n > 0
        claims_checked = 0
        obs_checked = 0
        for tendency in world.tendencies.values():
            for node in tendency.tree.all_nodes():
                if getattr(node, "artifact_digest", "") != DIGEST:
                    continue
                # Co-parented nodes appear in several trees; only some
                # tendencies hold the claim entry — the applier guards
                # that, so assert on the ones that exist.
                claim = tendency._node_to_claim.get(node.id)
                if claim is not None:
                    claims_checked += 1
                    assert list(claim.anchor[:N_DIMS]) == head
                    # Embedding tail untouched.
                    assert claim.anchor[N_DIMS + 3] == 0.5
                obs = world.observations.get(node.observation_id)
                if obs is not None:
                    obs_checked += 1
                    assert list(obs.coords[:N_DIMS]) == head
                    assert obs.coords[N_DIMS + 3] == 0.5
        assert claims_checked > 0
        assert obs_checked > 0


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
                                   embedding_dim=EMBED_DIM, review_snapshot=snap)
        r2 = federated_epoch_close(canonical_order(list(batches)),
                                   embedding_dim=EMBED_DIM, review_snapshot=snap)
        assert DIGEST in r1["tool_positions"]
        assert r1["tool_positions"][DIGEST]["head"][4] == pytest.approx(0.9)
        assert json.dumps(r1["tool_positions"]) == json.dumps(
            r2["tool_positions"])

    def test_next_close_reads_the_next_snapshot(self):
        batches = self._standard()
        r1 = federated_epoch_close(canonical_order(list(batches)),
                                   embedding_dim=EMBED_DIM,
                                   review_snapshot=_snap(90 * 50, 50))
        kp_c = Keypair.generate()
        b2 = self._batches([_receipt(1, "caller-2")], kp_c)
        # A later round finalized: the rating on chain moved.
        r2 = federated_epoch_close(
            canonical_order(b2),
            embedding_dim=EMBED_DIM,
            tool_registrations=r1["tool_registrations"],
            tool_positions=r1["tool_positions"],
            review_snapshot=_snap(90 * 50 - 90 * 50, 100),
        )
        assert r2["tool_positions"][DIGEST]["head"][4] == 0.0
        assert r2["tool_positions"][DIGEST]["mass"][4] == pytest.approx(0.1)
