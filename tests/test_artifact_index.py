"""ArtifactIndex — data-plane embedding index over blob-store artifacts.

Uses HashingEmbedder (deterministic, dependency-free) so tests are stable
and torch never enters the process. Covers add/get roundtrip, idempotence,
search ranking, persistence reload, rebuild-from-blobs, deterministic
tie-break, and in-memory mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nodes.common.blob_store import BlobStore
from nodes.common.world_model_substrate.artifact_index import ArtifactIndex
from nodes.common.world_model_substrate.usefulness_coords import HashingEmbedder


DIM = 64


@pytest.fixture
def blob_store(tmp_path: Path) -> BlobStore:
    return BlobStore(data_dir=str(tmp_path / "blobs"))


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=DIM)


def _work_unit(problem: str, resolution: str) -> dict:
    return {"problem": problem, "resolution": resolution, "outcome": {"ok": True}}


def test_add_and_get_roundtrip(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)
    payload = _work_unit("how to parse a date", "use strptime with a format string")
    digest = idx.add_artifact(payload)

    assert isinstance(digest, str) and len(digest) == 64  # sha256 hex
    assert idx.has(digest)
    assert blob_store.get_json(digest) == payload  # blob stored
    vec = idx.get(digest)
    assert vec is not None
    assert len(vec) == DIM


def test_idempotent_per_digest(tmp_path, blob_store, embedder):
    index_path = tmp_path / "index.jsonl"
    idx = ArtifactIndex(blob_store, embedder=embedder, index_path=index_path, dim=DIM)
    payload = _work_unit("compute a checksum", "hash the bytes with sha256")

    d1 = idx.add_artifact(payload)
    d2 = idx.add_artifact(dict(payload))  # equal payload -> same digest

    assert d1 == d2
    assert len(idx) == 1  # no duplicate memory entry

    # No duplicate jsonl row.
    rows = [ln for ln in index_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1


def test_search_ranking(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, dim=DIM)

    d_dates = idx.add_artifact(
        _work_unit(
            "parse a date string into a datetime object",
            "call datetime.strptime with the matching format code",
        )
    )
    idx.add_artifact(
        _work_unit(
            "resize an image to a thumbnail",
            "load the picture and scale its pixel dimensions down",
        )
    )
    idx.add_artifact(
        _work_unit(
            "sort a linked list in place",
            "reorder the nodes by pointer swaps without extra memory",
        )
    )

    results = idx.search("parse a date string into a datetime object", k=3)
    assert results, "expected non-empty results"
    top_digest, top_cos = results[0]
    assert top_digest == d_dates  # semantically matching artifact ranks first
    # Scores must be sorted descending.
    scores = [c for _, c in results]
    assert scores == sorted(scores, reverse=True)


def test_persistence_reload(tmp_path, blob_store, embedder):
    index_path = tmp_path / "index.jsonl"
    idx = ArtifactIndex(blob_store, embedder=embedder, index_path=index_path, dim=DIM)
    d1 = idx.add_artifact(_work_unit("problem one", "resolution one"))
    d2 = idx.add_artifact(_work_unit("problem two", "resolution two"))

    # Fresh index on the same path sees prior entries (eager load).
    reloaded = ArtifactIndex(blob_store, embedder=embedder, index_path=index_path, dim=DIM)
    assert reloaded.has(d1)
    assert reloaded.has(d2)
    assert len(reloaded) == 2
    # Embeddings survive the roundtrip.
    np.testing.assert_allclose(reloaded.get(d1), idx.get(d1))


def test_rebuild_from_blobs_after_index_deleted(tmp_path, blob_store, embedder):
    index_path = tmp_path / "index.jsonl"
    idx = ArtifactIndex(blob_store, embedder=embedder, index_path=index_path, dim=DIM)
    d_work = idx.add_artifact(_work_unit("build a widget", "assemble the parts"))
    d_text = idx.add_artifact({"text": "a generic note payload with some words"})
    # A non-artifact blob must be ignored on rebuild.
    blob_store.add_json({"meta": "not an artifact", "count": 3})

    # Delete the index file; the derived state must be rebuildable from blobs.
    index_path.unlink()
    fresh = ArtifactIndex(blob_store, embedder=embedder, index_path=index_path, dim=DIM)
    assert len(fresh) == 0  # nothing loaded

    count = fresh.rebuild()
    assert count == 2  # only the two artifact-shaped blobs
    assert fresh.has(d_work)
    assert fresh.has(d_text)
    # Rewritten jsonl exists and reloads.
    assert index_path.exists()
    reloaded = ArtifactIndex(blob_store, embedder=embedder, index_path=index_path, dim=DIM)
    assert len(reloaded) == 2


def test_deterministic_tie_break(blob_store):
    # Zero embedder -> every artifact has cosine 0.0, forcing a tie.
    class _ZeroEmbedder:
        def __call__(self, text: str):
            return tuple([0.0] * DIM)

    idx = ArtifactIndex(blob_store, embedder=_ZeroEmbedder(), dim=DIM)
    digests = [
        idx.add_artifact(_work_unit(f"problem {i}", f"resolution {i}"))
        for i in range(5)
    ]

    results = idx.search("anything", k=5)
    returned = [d for d, _ in results]
    assert returned == sorted(digests)  # tie-break by digest ascending


def test_in_memory_mode_no_persistence(blob_store, embedder):
    idx = ArtifactIndex(blob_store, embedder=embedder, index_path=None, dim=DIM)
    d = idx.add_artifact(_work_unit("in memory only", "no file is written"))

    assert idx.has(d)
    assert idx.index_path is None
    # Search still works without a backing file.
    results = idx.search("in memory only", k=1)
    assert results and results[0][0] == d
