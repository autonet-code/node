"""Artifact ingestion — the two-plane substrate's data plane.

Covers the digest that rides the ``SubClaimSprouted`` event and gets
attached to work-unit nodes, on both ingestion paths
(``train_world_model_on_usefulness`` and ``WorldService.submit_work_units``).

Determinism contract (docs/two_plane_inference.md): the digest is
serialized ONLY when non-empty, so events recorded before this field
existed — and every non-work-unit sprout — serialize byte-identically.
This keeps canonical-ordering batch hashes (and therefore epoch close)
unchanged. Tests here pin that back-compat guarantee.

Runs entirely on the hashing embedder (bit-deterministic, no subprocess).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nodes.common.blob_store import BlobStore
from nodes.common.world_model_substrate.artifact_index import ArtifactIndex
from nodes.common.world_model_substrate.events import (
    SubClaimSprouted,
    event_from_dict,
)
from nodes.common.world_model_substrate.outcomes import Outcome
from nodes.common.world_model_substrate.usefulness_coords import HashingEmbedder
from nodes.common.world_model_substrate.usefulness_training import (
    train_world_model_on_usefulness,
)
from nodes.common.world_service import WorldService


DIM = 32
EMBED_DIM = 16


@pytest.fixture(autouse=True)
def _hashing_embedder(monkeypatch):
    monkeypatch.setenv("ATN_USEFULNESS_EMBEDDER", "hashing")


@pytest.fixture
def blob_store(tmp_path: Path) -> BlobStore:
    return BlobStore(data_dir=str(tmp_path / "blobs"))


@pytest.fixture
def artifact_index(blob_store: BlobStore) -> ArtifactIndex:
    return ArtifactIndex(blob_store, embedder=HashingEmbedder(dim=DIM), dim=DIM)


# ---------------------------------------------------------------------------
# (a) + (b) Event serialization round-trip and back-compat
# ---------------------------------------------------------------------------


class TestEventSerialization:
    def test_roundtrip_with_digest(self):
        digest = "a" * 64
        ev = SubClaimSprouted(seq=1, author_agent="0xA", artifact_digest=digest)
        d = ev.to_dict()
        assert d["artifact_digest"] == digest
        restored = event_from_dict(d)
        assert restored.artifact_digest == digest

    def test_to_dict_omits_key_when_empty(self):
        ev = SubClaimSprouted(seq=1, author_agent="0xA")  # default ""
        d = ev.to_dict()
        assert "artifact_digest" not in d
        # Round-trip still works and lands on the empty default.
        restored = event_from_dict(d)
        assert restored.artifact_digest == ""

    def test_old_format_dict_without_key_deserializes(self):
        # An event dict recorded by an older daemon: no artifact_digest.
        old = {
            "kind": "sub_claim_sprouted",
            "seq": 3,
            "author_agent": "0xLegacy",
            "tendency_id": "good_resolution",
            "parent_id": "root_x",
            "node_id": "node_y",
            "position": "pro",
            "coords": [0.0, 1.0],
            "polarity_axis": [1.0, 0.0],
            "content": "legacy claim",
            "observation_id": "obs_z",
        }
        ev = event_from_dict(old)  # must not raise
        assert isinstance(ev, SubClaimSprouted)
        assert ev.artifact_digest == ""
        # And it re-serializes without introducing the key (byte parity).
        assert "artifact_digest" not in ev.to_dict()


# ---------------------------------------------------------------------------
# (c) + (d) usefulness_training ingestion path
# ---------------------------------------------------------------------------


def _work_units():
    return [
        (
            "parse a date string into a datetime",
            "call datetime.strptime with the matching format code",
            Outcome(accepted=1.0, kept=1.0),
        ),
        (
            "resize an image to a thumbnail",
            "load the picture and scale its pixel dimensions down",
            Outcome(accepted=-1.0),
        ),
    ]


def _sprout_events(contribution):
    return [
        e for e in contribution["events"]
        if e.get("kind") == "sub_claim_sprouted"
    ]


class TestUsefulnessTrainingIngestion:
    def test_digests_attached_and_blobs_exist(self, artifact_index, blob_store):
        contribution, _ = train_world_model_on_usefulness(
            _work_units(),
            dim=DIM,
            agent_id="0xTrainer",
            embedder=HashingEmbedder(dim=DIM),
            artifact_index=artifact_index,
        )
        sprouts = _sprout_events(contribution)
        with_digest = [e for e in sprouts if e.get("artifact_digest")]
        assert with_digest, "expected work-unit sprouts to carry digests"
        # A work-unit node can be emitted under more than one tendency
        # (co-parenting), so count DISTINCT digests: one per work unit.
        distinct = {e["artifact_digest"] for e in with_digest}
        assert len(distinct) == len(_work_units())
        for digest in distinct:
            assert len(digest) == 64  # sha256 hex
            assert artifact_index.has(digest)
            payload = blob_store.get_json(digest)
            assert payload is not None
            assert "problem" in payload
            assert "resolution" in payload
            assert isinstance(payload["outcome"], dict)

    def test_none_index_is_byte_identical(self):
        # Same work units, same agent/embedder, once with and once
        # without the index. The emitted event dicts must be identical,
        # and the no-index run must carry no artifact_digest key at all.
        units = _work_units()

        contrib_none, _ = train_world_model_on_usefulness(
            units, dim=DIM, agent_id="0xT", embedder=HashingEmbedder(dim=DIM),
        )
        for ev in contrib_none["events"]:
            assert "artifact_digest" not in ev

        # A second no-index run must reproduce the exact same event dicts
        # (determinism unaffected by the new code path).
        contrib_none2, _ = train_world_model_on_usefulness(
            units, dim=DIM, agent_id="0xT", embedder=HashingEmbedder(dim=DIM),
        )
        assert contrib_none["events"] == contrib_none2["events"]


# ---------------------------------------------------------------------------
# (e) WorldService construction + submit_work_units
# ---------------------------------------------------------------------------


class TestWorldServiceIngestion:
    def test_construction_creates_index_and_blob_store(self, tmp_path):
        svc = WorldService(
            "0xArtifactSvc",
            data_root=tmp_path,
            embedding_dim=EMBED_DIM,
            snapshot_every_n_events=10_000,
            snapshot_every_seconds=10_000.0,
        )
        assert isinstance(svc._blob_store, BlobStore)
        assert isinstance(svc._artifact_index, ArtifactIndex)
        state_dir = svc._persistence._dir
        assert (state_dir / "blobs").is_dir()
        assert svc._artifact_index.index_path == state_dir / "artifact_index.jsonl"

    def test_submit_work_units_attaches_digests(self, tmp_path):
        svc = WorldService(
            "0xArtifactSvc2",
            data_root=tmp_path,
            embedding_dim=EMBED_DIM,
            snapshot_every_n_events=10_000,
            snapshot_every_seconds=10_000.0,
        )
        problem = "What is the cache TTL?"
        resolution = (
            "The prompt cache TTL is five minutes; sleep past 300s and the "
            "next wake-up reads context uncached."
        )
        receipt = svc.submit_work_units(
            [(problem, resolution, Outcome(accepted=1.0, kept=1.0))],
            agent_id="0xAuthor",
            embedder=HashingEmbedder(dim=EMBED_DIM),
        )
        assert receipt["units_processed"] == 1

        # The work-unit node in the live world carries the digest.
        wu_node = None
        for tendency in svc._world.tendencies.values():
            for node in tendency.tree.all_nodes():
                if (node.observation_id or "").startswith("wu_"):
                    wu_node = node
                    break
        assert wu_node is not None
        digest = getattr(wu_node, "artifact_digest", "")
        assert digest and len(digest) == 64
        assert svc._artifact_index.has(digest)

        # The blob holds the full payload.
        payload = svc._blob_store.get_json(digest)
        assert payload is not None
        assert payload["problem"] == problem
        assert payload["resolution"] == resolution
        assert isinstance(payload["outcome"], dict)

    def test_shared_blob_store_is_used(self, tmp_path):
        # An injected blob store is honored (not overridden).
        shared = BlobStore(data_dir=str(tmp_path / "shared_blobs"))
        svc = WorldService(
            "0xArtifactSvc3",
            data_root=tmp_path,
            embedding_dim=EMBED_DIM,
            blob_store=shared,
            snapshot_every_n_events=10_000,
            snapshot_every_seconds=10_000.0,
        )
        assert svc._blob_store is shared
        assert svc._artifact_index.blob_store is shared


# ---------------------------------------------------------------------------
# Replay: the digest survives apply_events onto a replica world
# ---------------------------------------------------------------------------


class TestReplayCarriesDigest:
    def test_apply_events_reattaches_digest(self, artifact_index):
        from nodes.common.world_model_substrate.aggregate import apply_events
        from nodes.common.world_model_substrate.usefulness_training import (
            build_usefulness_world,
        )

        contribution, _ = train_world_model_on_usefulness(
            _work_units(),
            dim=DIM,
            agent_id="0xTrainer",
            embedder=HashingEmbedder(dim=DIM),
            artifact_index=artifact_index,
        )
        emitted = {
            e["artifact_digest"]
            for e in _sprout_events(contribution)
            if e.get("artifact_digest")
        }
        assert emitted

        # A replica daemon replays the gossiped events onto its own world
        # and must end up with digest-bearing nodes, otherwise standings
        # are silently zero everywhere off the authoring daemon.
        replica = build_usefulness_world(dim=DIM)
        apply_events(replica, contribution["events"])
        replayed = {
            getattr(node, "artifact_digest", "")
            for tendency in replica.tendencies.values()
            for node in tendency.tree.all_nodes()
        }
        assert emitted <= replayed
