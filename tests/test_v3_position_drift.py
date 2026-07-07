"""v3 position drift (docs/tool_substrate.md, Decision 2026-07-08).

A tool's charter head is the per-axis mint-weighted running centroid of
review scores: prior = zero head with 1.0 damped unit of mass per axis;
`head' = (mass·head + Σ w·mean) / (mass + Σ w)` with w = log1p(caller's
axis review count), same exclusions as the usage damper. Covers: entry,
drift math, partial-axis safety, copy-through, cross-epoch continuity,
exclusions, world application (claim anchors + observation coords), and
bit-identical replay.
"""

from __future__ import annotations

import json
import math
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


class TestDriftMath:
    def test_new_tool_enters_at_neutral_prior(self):
        world = _world_with_registration()
        out = compute_tool_mint(world, [], registrations=REGS)
        pos = out["positions_next"][DIGEST]
        assert pos["head"] == [0.0] * N_DIMS
        assert pos["mass"] == [1.0] * N_DIMS

    def test_single_review_drifts_scored_axis_only(self):
        world = _world_with_registration()
        out = compute_tool_mint(
            world,
            [_receipt(1, "caller-1", axes={"correctness": 0.8})],
            registrations=REGS,
        )
        pos = out["positions_next"][DIGEST]
        w = math.log1p(1)
        # correctness is axis_index 4 (adapter CHARTER order).
        expected = (1.0 * 0.0 + w * 0.8) / (1.0 + w)
        assert pos["head"][4] == pytest.approx(expected, abs=1e-9)
        assert pos["mass"][4] == pytest.approx(1.0 + w, abs=1e-9)
        # Unscored axes untouched (partial scoring is safe).
        for i in range(N_DIMS):
            if i == 4:
                continue
            assert pos["head"][i] == 0.0
            assert pos["mass"][i] == 1.0

    def test_heavily_reviewed_tool_has_inertia(self):
        world = _world_with_registration()
        carried = {DIGEST: {"head": [0, 0, 0, 0, 0.9, 0],
                            "mass": [1, 1, 1, 1, 100.0, 1]}}
        out = compute_tool_mint(
            world,
            [_receipt(1, "caller-1", axes={"correctness": -1.0})],
            registrations=REGS, positions=carried,
        )
        head = out["positions_next"][DIGEST]["head"]
        # One negative review barely moves 100 units of accumulated mass.
        assert head[4] > 0.85

    def test_copy_through_without_reviews(self):
        world = _world_with_registration()
        carried = {DIGEST: {"head": [0.1] * N_DIMS, "mass": [5.0] * N_DIMS}}
        out = compute_tool_mint(
            world, [_receipt(1, "caller-1")],   # usage, no axes
            registrations=REGS, positions=carried,
        )
        pos = out["positions_next"][DIGEST]
        assert pos["head"] == [0.1] * N_DIMS
        assert pos["mass"] == [5.0] * N_DIMS

    def test_author_and_same_owner_reviews_excluded(self):
        world = _world_with_registration()
        out = compute_tool_mint(
            world,
            [
                _receipt(1, AUTHOR, axes={"correctness": 1.0}),
                _receipt(2, "sibling", axes={"correctness": 1.0}),
            ],
            registrations=REGS,
            agent_owner_map={AUTHOR: "0xOwner", "sibling": "0xOwner"},
        )
        pos = out["positions_next"][DIGEST]
        assert pos["head"] == [0.0] * N_DIMS   # nothing counted

    def test_cross_epoch_continuity(self):
        world = _world_with_registration()
        out1 = compute_tool_mint(
            world, [_receipt(1, "caller-1", axes={"correctness": 0.8})],
            registrations=REGS,
        )
        out2 = compute_tool_mint(
            world, [_receipt(1, "caller-2", axes={"correctness": -0.4})],
            registrations=REGS,
            positions=out1["positions_next"],
        )
        w = math.log1p(1)
        h1 = (w * 0.8) / (1.0 + w)
        m1 = 1.0 + w
        expected = (m1 * h1 + w * -0.4) / (m1 + w)
        assert out2["positions_next"][DIGEST]["head"][4] == pytest.approx(
            expected, abs=1e-8)


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
        r1 = federated_epoch_close(canonical_order(list(batches)),
                                   embedding_dim=EMBED_DIM)
        r2 = federated_epoch_close(canonical_order(list(batches)),
                                   embedding_dim=EMBED_DIM)
        assert DIGEST in r1["tool_positions"]
        assert r1["tool_positions"][DIGEST]["head"][4] > 0.0
        assert json.dumps(r1["tool_positions"]) == json.dumps(
            r2["tool_positions"])

    def test_carry_threads_across_closes(self):
        batches = self._standard()
        r1 = federated_epoch_close(canonical_order(list(batches)),
                                   embedding_dim=EMBED_DIM)
        # Epoch 2: registration carried, one more review.
        kp_c = Keypair.generate()
        b2 = self._batches(
            [_receipt(1, "caller-2", axes={"correctness": -0.9})], kp_c)
        r2 = federated_epoch_close(
            canonical_order(b2),
            embedding_dim=EMBED_DIM,
            tool_registrations=r1["tool_registrations"],
            tool_positions=r1["tool_positions"],
        )
        h1 = r1["tool_positions"][DIGEST]["head"][4]
        h2 = r2["tool_positions"][DIGEST]["head"][4]
        assert h2 < h1                       # dragged down by the -0.9
        m1 = r1["tool_positions"][DIGEST]["mass"][4]
        m2 = r2["tool_positions"][DIGEST]["mass"][4]
        assert m2 == pytest.approx(m1 + math.log1p(1), abs=1e-8)
