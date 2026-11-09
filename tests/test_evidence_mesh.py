"""A malware flag's evidence travels on the mesh (Decision 2026-11-09,
docs/p2p_networking.md "Blobs"): the reviewer's daemon keeps the blob and
provides its hash in the DHT; a daemon that is not connected to it finds
it with find_providers and reads the evidence through fetch_evidence,
dialling the reviewer's daemon through its relay circuit. The relay takes
the provider record only, not the evidence.

Real localhost hosts run in their own trio thread, as in the daemon; the
tool stores run on asyncio in the test's thread, wired as the daemon wires
them (atn/autonet_service.py).

Run: python -m pytest tests/test_evidence_mesh.py -v
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes.common.blob_resolver import LibP2PBlobResolver
from tests.atn.test_tool_store import _make_runtime
from tests.test_search_model_mesh import _Mesh

MANIFEST = {"name": "exfil_tool", "code_digest": "ab" * 32, "author": "0x" + "11" * 20}
TOOL = "cd" * 32


def test_flag_evidence_is_provided_and_fetched_through_the_dht(tmp_path):
    reviewer = _make_runtime(tmp_path / "reviewer").tool_store
    reader = _make_runtime(tmp_path / "reader").tool_store

    with _Mesh() as mesh:
        holder, relay = mesh.hosts["holder"], mesh.hosts["relay"]
        # The reviewer's daemon: its blob store served, the announcer on.
        reviewer._blob_store().set_p2p_host(holder)
        reviewer.blob_announcer = (
            lambda digests, _h=holder: _h.provide_blobs_blocking(digests, timeout=15.0))

        async def flag() -> str:
            # In a running loop, as in the daemon: the announce is not awaited.
            return reviewer.store_evidence(
                TOOL, MANIFEST, "user",
                "tool.py line 3 posts ~/.ssh/id_rsa to evil.example")
        evidence_hash = asyncio.run(flag())
        assert len(evidence_hash) == 64
        deadline = time.time() + 45
        while evidence_hash not in holder._provided_at and time.time() < deadline:
            time.sleep(0.25)
        assert evidence_hash in holder._provided_at, "no provider record placed"

        # Another daemon joins through the relay only, wired like the daemon.
        fetcher = mesh.join()
        assert holder.peer_id not in fetcher.get_connected_peers()
        resolver = LibP2PBlobResolver(fetcher)

        async def _fetch(digest, _r=resolver):
            return await asyncio.to_thread(_r.get, digest)
        reader.blob_fetcher = _fetch
        got = asyncio.run(reader.fetch_evidence(evidence_hash))

        assert got["found"] is True, got
        assert got["evidence"]["kind"] == "malware_evidence"
        assert got["evidence"]["manifest_digest"] == TOOL
        assert "line 3" in got["evidence"]["evidence"]
        # Found through the provider record, not by asking a neighbour: the
        # relay, the fetcher's only neighbour, holds no copy.
        assert fetcher.last_fetch_source == "dht"
        assert holder.peer_id in fetcher.connection_kinds()
        server = getattr(relay, "_autonet_blob_server", None)
        assert server is None or server.lookup(evidence_hash) is None


def test_an_offline_reviewer_s_evidence_is_not_found(tmp_path):
    """Keeping the evidence available is the reviewer's job: with no
    holder reachable, fetch_evidence says so."""
    reader = _make_runtime(tmp_path / "reader").tool_store

    async def nobody(_digest):
        return None
    reader.blob_fetcher = nobody
    got = asyncio.run(reader.fetch_evidence("ef" * 32))
    assert got["found"] is False and "reviewer" in got["note"]
