"""Pruning (Decision 2026-11-10): the close lists every tool with an
uncleared malware verdict, every tool built on one and every refused
registration; daemons drop them from the library, delete their blobs except
what the household installed, and never serve or fetch them. A DAO
clearing makes a tool fetchable again."""

from __future__ import annotations

import pytest

from nodes.common import pruned_tools
from nodes.common.blob_resolver import HostBlobServer
from nodes.common.blob_store import BlobStore
from nodes.common.library.tool_manifest import build_tool_manifest
from nodes.common.library_service import LibraryService


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("ATN_USEFULNESS_EMBEDDER", "hashing")
    pruned_tools.replace([])
    yield
    pruned_tools.replace([])


def _lib(tmp_path):
    blobs = BlobStore(data_dir=str(tmp_path / "blobs"))
    return LibraryService("rpb_prune", data_root=tmp_path / "world", blob_store=blobs), blobs


def _tool(lib, blobs, name):
    code = blobs.add_bytes(f"print('{name}')".encode())
    manifest = build_tool_manifest(
        name=name, description=f"{name} for the pruning test.",
        input_schema={"type": "object", "properties": {}}, author="0xA",
        trust_class="pinned", code_digest=code, created_ts=1780531200)
    digest = lib.index_remote_manifest(manifest)
    return digest, code, manifest


def test_pruning_drops_the_index_deletes_blobs_and_refuses_serving(tmp_path):
    lib, blobs = _lib(tmp_path)
    bad, code, manifest = _tool(lib, blobs, "bad_tool")
    good, good_code, _ = _tool(lib, blobs, "good_tool")
    out = lib.apply_pruned([bad])
    assert out["new"] == 1 and out["deleted"] == 2
    assert bad not in lib.artifact_digests() and good in lib.artifact_digests()
    assert not blobs.has(bad) and not blobs.has(code) and blobs.has(good_code)
    assert pruned_tools.is_pruned(bad) and pruned_tools.is_pruned(code)
    # Never fetched again, never re-indexed from gossip.
    assert blobs.get_bytes(bad) is None
    assert lib.index_remote_manifest(manifest) is None
    lib.submit_events([{"kind": "tool_registered", "seq": 1, "author_agent": "0xA",
                        "artifact_digest": bad, "manifest_meta": {"author": "0xA"}}])
    assert bad not in lib.take_pending_remote_manifests()


def test_a_household_keeps_what_it_installed_but_never_serves_it(tmp_path):
    lib, blobs = _lib(tmp_path)
    bad, code, _ = _tool(lib, blobs, "installed_tool")
    keep = lambda d: d in {bad, code}  # noqa: E731
    pruned_tools.add_keeper(keep)
    try:
        lib.apply_pruned([bad])
        assert blobs.has(bad) and blobs.has(code)          # its own copy
        assert blobs.get_bytes(code) is not None           # still runs here
        assert blobs.serve_local(code) is None             # never to peers
        server = HostBlobServer(type("H", (), {"set_blob_handler": lambda self, f: None})())
        server.add_source(blobs.get_bytes_local)
        assert server.lookup(code) is None
    finally:
        pruned_tools._keepers.remove(keep)


def test_a_dao_clearing_unprunes_and_a_restart_remembers(tmp_path):
    lib, blobs = _lib(tmp_path)
    bad, code, _ = _tool(lib, blobs, "cleared_tool")
    lib.apply_pruned([bad])
    again, _ = _lib(tmp_path)                      # restart: same state dir
    assert again.pruned_tool_digests() == [bad] and pruned_tools.is_pruned(code)
    again.apply_pruned([])
    assert not pruned_tools.is_pruned(bad) and not pruned_tools.is_pruned(code)


def test_the_close_result_carries_the_list_to_the_library(tmp_path):
    lib, blobs = _lib(tmp_path)
    bad, _, _ = _tool(lib, blobs, "closed_tool")
    lib.note_federated_close({"pruned": [bad], "tool_registrations": {},
                              "tool_ratings": {}}, epoch_id="e1")
    assert lib.pruned_tool_digests() == [bad]
