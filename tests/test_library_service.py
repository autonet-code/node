"""LibraryService: the tool library's state dir, search index, discovery
queue, epoch tick and the display copy of each federated close.

Runs on the hashing embedder (bit-deterministic, no model download).
"""

from __future__ import annotations

import pytest

from nodes.common.blob_store import BlobStore
from nodes.common.library.artifact_index import ArtifactIndex
from nodes.common.library.tool_manifest import build_tool_manifest
from nodes.common.library_service import LibraryService


@pytest.fixture(autouse=True)
def _hashing_embedder(monkeypatch):
    monkeypatch.setenv("ATN_USEFULNESS_EMBEDDER", "hashing")


def _manifest(name: str, *, author: str = "0xAuthor", dependencies=None):
    return build_tool_manifest(
        name=name,
        description=f"{name}: does a thing for the library test.",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        author=author,
        trust_class="pinned",
        code_digest="c" * 64,
        dependencies=dependencies,
        created_ts=1780531200,
    )


def _reg_event(digest: str) -> dict:
    return {
        "kind": "tool_registered",
        "seq": 1,
        "author_agent": "0xPeer",
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": "pinned", "author": "0xPeer"},
    }


class TestConstruction:
    def test_creates_index_and_blob_store(self, tmp_path):
        svc = LibraryService("0xLibrarySvc", data_root=tmp_path)
        assert isinstance(svc._blob_store, BlobStore)
        assert isinstance(svc._artifact_index, ArtifactIndex)
        assert (svc.state_dir / "blobs").is_dir()
        assert svc._artifact_index.index_path == svc.state_dir / "artifact_index.jsonl"

    def test_shared_blob_store_is_used(self, tmp_path):
        shared = BlobStore(data_dir=str(tmp_path / "shared_blobs"))
        svc = LibraryService("0xLibrarySvc2", data_root=tmp_path, blob_store=shared)
        assert svc._blob_store is shared
        assert svc._artifact_index.blob_store is shared


class TestRegistration:
    def test_publish_indexes_and_emits_one_event(self, tmp_path):
        svc = LibraryService("0xReg1", data_root=tmp_path)
        sent = []
        svc.subscribe_local_events(sent.extend)
        svc.open_epoch("e1")
        manifest = _manifest("dep_tool", dependencies=["ab" * 32])
        receipt = svc.submit_tool_manifest(manifest, agent_id="0xAgent")
        digest = receipt["manifest_digest"]
        assert svc._artifact_index.has(digest)
        assert svc.blob_json(digest)["name"] == "dep_tool"
        assert sent == [{
            "kind": "tool_registered", "seq": 1, "author_agent": "0xAgent",
            "artifact_digest": digest,
            "manifest_meta": {"trust_class": "pinned", "author": "0xAuthor",
                              "deps": "ab" * 32},
        }]
        assert svc.epoch_status()["buffered_events"] == 1
        assert svc.close_epoch()["n_events"] == 1

    def test_invalid_manifest_is_refused(self, tmp_path):
        svc = LibraryService("0xReg2", data_root=tmp_path)
        assert "error" in svc.submit_tool_manifest({"kind": "tool_manifest"})


class TestRemoteManifestDiscovery:
    def test_remote_registration_queues_its_digest(self, tmp_path):
        svc = LibraryService("0xDiscover1", data_root=tmp_path)
        digest = "b" * 64
        svc.submit_events([_reg_event(digest)], origin="remote")
        assert svc.take_pending_remote_manifests() == [digest]
        assert svc.take_pending_remote_manifests() == []

    def test_local_origin_is_not_queued(self, tmp_path):
        svc = LibraryService("0xDiscover2", data_root=tmp_path)
        svc.submit_events([_reg_event("c" * 64)], origin="local")
        assert svc.take_pending_remote_manifests() == []

    def test_indexing_a_fetched_manifest_makes_it_searchable(self, tmp_path):
        svc = LibraryService("0xDiscover3", data_root=tmp_path)
        manifest = {
            "kind": "tool_manifest",
            "name": "summarize_csv",
            "description": "summarize a csv file into per-column statistics",
            "author": "0xPeer",
            "trust_class": "pinned",
            "input_schema": {"type": "object", "properties": {}},
            "code_digest": "d" * 64,
        }
        digest = svc.index_remote_manifest(manifest)
        assert digest and digest in svc.artifact_digests()
        hits = svc._artifact_index.search("csv column statistics", k=3)
        assert digest in [d for d, _ in hits]

    def test_non_manifest_payload_is_rejected(self, tmp_path):
        svc = LibraryService("0xDiscover4", data_root=tmp_path)
        assert svc.index_remote_manifest({"problem": "x", "resolution": "y"}) is None

    def test_already_indexed_digest_is_not_queued(self, tmp_path):
        svc = LibraryService("0xDiscover5", data_root=tmp_path)
        digest = svc.index_remote_manifest({
            "kind": "tool_manifest", "name": "known_tool",
            "description": "a tool this daemon already indexed",
            "author": "0xPeer", "trust_class": "pinned",
            "input_schema": {"type": "object", "properties": {}},
            "code_digest": "e" * 64,
        })
        svc.submit_events([_reg_event(digest)], origin="remote")
        assert svc.take_pending_remote_manifests() == []


class TestEpochTick:
    def test_candle_cutoff_rolls_late_events_forward(self, tmp_path):
        import time
        svc = LibraryService("0xTick1", data_root=tmp_path)
        svc.open_epoch("e1")
        svc.submit_events([_reg_event("a" * 64)], origin="remote")
        cut = time.time()
        time.sleep(0.01)
        svc.submit_events([_reg_event("b" * 64)], origin="remote")
        rec = svc.close_epoch(cutoff_ts=cut)
        assert rec["n_events"] == 1 and rec["events_rolled_forward"] == 1
        svc.open_epoch("e2")
        assert svc.epoch_status()["buffered_events"] == 1

    def test_history_survives_restart(self, tmp_path):
        svc = LibraryService("0xTick2", data_root=tmp_path)
        svc.open_epoch("e1")
        svc.close_epoch()
        again = LibraryService("0xTick2", data_root=tmp_path)
        assert [r["epoch_id"] for r in again.epoch_history] == ["e1"]
        assert again.epochs_recorded == 1
