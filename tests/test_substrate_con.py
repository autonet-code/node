"""Targeted CON primitive + claim-on-the-wire contract.

Covers the two changes that make substrate entries disputable:

  1. ``WorldService.submit_con`` attaches a CON child *under* a
     specific live node (the missing reply-to), with an authored
     unit-weight post that replays deterministically
     (``author_post`` on the sprout event).
  2. Work units ship their condensed claim as the event label, and
     the coordinate tail derives from that claim alone — so any peer
     can recompute the embedding from the event and prove fabricated
     coords (``verify_claim_coords``).

Runs entirely on the hashing embedder (bit-deterministic, no
subprocess).
"""

from __future__ import annotations

import json

import pytest

from nodes.common.world_service import (
    CLAIM_MAX_CHARS,
    WorldService,
    _claim_text,
)
from nodes.common.world_model_substrate.aggregate import apply_events
from nodes.common.world_model_substrate.adapter import N_DIMS, build_charter_world
from nodes.common.world_model_substrate.usefulness_coords import (
    HashingEmbedder,
    verify_claim_coords,
)


EMBED_DIM = 16


@pytest.fixture(autouse=True)
def _hashing_embedder(monkeypatch):
    monkeypatch.setenv("ATN_USEFULNESS_EMBEDDER", "hashing")


@pytest.fixture
def service(tmp_path):
    svc = WorldService(
        "0xTestSubstrate",
        data_root=tmp_path,
        embedding_dim=EMBED_DIM,
        snapshot_every_n_events=10_000,
        snapshot_every_seconds=10_000.0,
    )
    return svc


def _submit_unit(svc, problem="What is the cache TTL?",
                 resolution="The prompt cache TTL is five minutes; sleep past "
                            "300s and the next wake-up reads context uncached.",
                 agent="0xAuthor"):
    receipt = svc.submit_work_units(
        [(problem, resolution, None)],
        agent_id=agent,
        embedder=HashingEmbedder(dim=EMBED_DIM),
    )
    assert receipt["units_processed"] == 1
    return receipt


def _wu_node(svc):
    """Find the explicit rootless work-unit node in the live world."""
    for tendency in svc._world.tendencies.values():
        for node in tendency.tree.all_nodes():
            if (node.observation_id or "").startswith("wu_"):
                return node
    raise AssertionError("work-unit node not found")


# ---------------------------------------------------------------------------
# Claim on the wire
# ---------------------------------------------------------------------------


class TestClaimOnTheWire:
    def test_label_is_condensed_claim_not_teaser(self, service):
        long_problem = "P" * 500
        long_resolution = "R" * 500
        _submit_unit(service, problem=long_problem, resolution=long_resolution)
        node = _wu_node(service)
        obs = service._world.observations.get(node.observation_id)
        assert obs is not None
        assert len(obs.label) <= CLAIM_MAX_CHARS
        # The claim carries the resolution head, not just the problem.
        assert "R" in obs.label and "::" in obs.label

    def test_coords_derive_from_claim_and_verify(self, service):
        _submit_unit(service)
        node = _wu_node(service)
        obs = service._world.observations[node.observation_id]
        result = verify_claim_coords(
            obs.coords, obs.label, head_dims=N_DIMS, dim=EMBED_DIM,
        )
        assert result["valid"], result
        assert result["backend"] == "hashing"
        assert result["cosine"] == pytest.approx(1.0)

    def test_fabricated_coords_fail_verification(self, service):
        _submit_unit(service)
        node = _wu_node(service)
        obs = service._world.observations[node.observation_id]
        fabricated = tuple(obs.coords[:N_DIMS]) + tuple(
            0.5 if i % 2 == 0 else 0.0 for i in range(EMBED_DIM)
        )
        result = verify_claim_coords(
            fabricated, obs.label, head_dims=N_DIMS, dim=EMBED_DIM,
        )
        assert not result["valid"], result

    def test_charter_head_is_zeros_at_n_dims(self, service):
        """Alignment is not self-graded: the head is N_DIMS zeros (the
        old path shipped a 4-zero head that misaligned the tail into
        charter dims 4-5)."""
        _submit_unit(service)
        node = _wu_node(service)
        obs = service._world.observations[node.observation_id]
        assert len(obs.coords) == N_DIMS + EMBED_DIM
        assert all(c == 0.0 for c in obs.coords[:N_DIMS])

    def test_claim_text_condensation(self):
        claim = _claim_text("  why   so\nslow ", "because of   replay " * 50)
        assert len(claim) <= CLAIM_MAX_CHARS
        assert claim.startswith("why so slow :: because of replay")
        assert _claim_text("", "answer only") == "answer only"


# ---------------------------------------------------------------------------
# Targeted CON
# ---------------------------------------------------------------------------


class TestTargetedCon:
    def test_con_attaches_under_target_and_drags_score(self, service):
        _submit_unit(service)
        target = _wu_node(service)
        # Give the target organic standing so the drag is observable.
        target.add_post("0xSupporter")
        target.invalidate_cache()
        score_before = target.net_score

        receipt = service.submit_con(
            target.id,
            "The TTL figure is wrong: the cache holds for 1h, not 5m.",
            agent_id="0xDisputer",
        )

        assert receipt["target_node_id"] == target.id
        con_node = None
        for tendency in service._world.tendencies.values():
            n = tendency.tree.get_node(receipt["con_node_id"])
            if n is not None:
                con_node = n
                break
        assert con_node is not None
        assert con_node in target.con_children
        # Authored post gives the dispute standing of 1 immediately.
        assert any(s.agent_id == "0xDisputer" for s in con_node.stakes)
        target.invalidate_cache()
        assert target.net_score <= score_before - 1

    def test_con_requires_existing_target(self, service):
        with pytest.raises(ValueError, match="unknown target node"):
            service.submit_con("n_doesnotexist", "bogus", agent_id="0xD")

    def test_con_rejects_charter_roots(self, service):
        root_id = next(iter(service._world.tendencies.values())).tree.root_node.id
        with pytest.raises(ValueError, match="charter root"):
            service.submit_con(root_id, "dispute the axiom", agent_id="0xD")

    def test_con_claim_is_capped_and_required(self, service):
        _submit_unit(service)
        target = _wu_node(service)
        with pytest.raises(ValueError, match="claim text is required"):
            service.submit_con(target.id, "   ", agent_id="0xD")
        receipt = service.submit_con(target.id, "x " * 400, agent_id="0xD")
        con = None
        for t in service._world.tendencies.values():
            con = t.tree.get_node(receipt["con_node_id"]) or con
        assert con is not None and len(con.content) <= CLAIM_MAX_CHARS


# ---------------------------------------------------------------------------
# Federated replay parity
# ---------------------------------------------------------------------------


class TestReplayParity:
    def test_causal_log_replays_con_identically(self, service, tmp_path):
        """A rejoining peer replaying the persisted causal log (full
        replay, markers honored — the production rail) reproduces the
        targeted CON: same parent edge, same authored post, same
        net_score on the target.

        Marker-segmented replay means the CON's sprout batch does NOT
        contain the target's sprout event, so this exercises
        ``resolve_parent``'s live-node branch (cross-batch targeting).
        """
        from nodes.common.world_persistence import (
            PersistenceConfig,
            WorldPersistence,
        )

        _submit_unit(service)
        target = _wu_node(service)
        service.submit_con(
            target.id, "Disputed: wrong constant.", agent_id="0xDisputer",
        )
        target.invalidate_cache()
        live_target_score = target.net_score

        # Force the full-replay path (no checkpoint fast path).
        service._persistence.checkpoint_path.unlink(missing_ok=True)
        peer_persistence = WorldPersistence(PersistenceConfig(
            rpb_address=service.rpb_address,
            data_root=tmp_path,
            embedding_dim=EMBED_DIM,
            bandwidth=service.bandwidth,
        ))
        restored = peer_persistence.try_restore()
        assert restored is not None and not restored.from_checkpoint
        peer = restored.world

        peer_target = None
        for tendency in peer.tendencies.values():
            n = tendency.tree.get_node(target.id)
            if n is not None:
                peer_target = n
                break
        assert peer_target is not None, "target node missing on peer"
        # CON child exists under the same deterministic parent.
        assert any(
            any(s.agent_id == "0xDisputer" for s in c.stakes)
            for c in peer_target.con_children
        ), "authored CON post missing on peer"
        peer_target.invalidate_cache()
        assert peer_target.net_score == pytest.approx(live_target_score)

    def test_author_post_survives_apply_stakes_rounds(self, service):
        """Tendency rounds strip and re-post tendency stakes each
        round; the disputer's authored post must persist."""
        _submit_unit(service)
        target = _wu_node(service)
        receipt = service.submit_con(target.id, "wrong", agent_id="0xDisputer")
        service._world.apply_stakes()
        con = None
        for t in service._world.tendencies.values():
            con = t.tree.get_node(receipt["con_node_id"]) or con
        assert any(s.agent_id == "0xDisputer" for s in con.stakes)
