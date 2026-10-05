"""Phase 10.3d: blob resolver tests.

InMemoryBlobResolver is exercised indirectly across the codebase;
this file focuses on LibP2PBlobResolver's local-side behavior:

  - Local put → local get fast path.
  - Miss with no peers known → None (not an exception).
  - Wired blob handler serves from local store (this is what peers
    hit when they fetch from us).

Cross-process peer-fetch isn't covered here — that needs a real
two-daemon harness which lives in Phase 10.4/5.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nodes.common.authoritative_encoding import cid_for_blob
from nodes.common.blob_resolver import LibP2PBlobResolver


def _fake_host():
    """A bare MagicMock host with set_blob_handler + known_capabilities."""
    host = MagicMock()
    host._trio_token = None
    host.known_capabilities = {}
    return host


def test_put_then_local_get():
    resolver = LibP2PBlobResolver(_fake_host())
    blob = b"some test blob bytes"
    cid = resolver.put(blob)
    assert cid == cid_for_blob(blob)
    assert resolver.get(cid) == blob


def test_miss_no_peers_returns_none():
    resolver = LibP2PBlobResolver(_fake_host())
    assert resolver.get(cid_for_blob(b"never stored")) is None


def test_set_blob_handler_invoked_on_construction():
    """The resolver registers itself as the host's blob handler so
    peers hitting /autonet/blob/1.0.0 can fetch from our local store."""
    host = _fake_host()
    LibP2PBlobResolver(host)
    host.set_blob_handler.assert_called_once()
    # The registered handler is async — verify it serves from local
    # by invoking it with an unknown cid (should return None) and
    # then putting a blob and re-invoking.
    handler = host.set_blob_handler.call_args.args[0]
    import trio
    # No blob stored yet → handler returns None.
    assert trio.run(handler, "bogus_cid") is None


def test_handler_serves_from_local_store():
    host = _fake_host()
    resolver = LibP2PBlobResolver(host)
    handler = host.set_blob_handler.call_args.args[0]
    blob = b"served-via-handler"
    cid = resolver.put(blob)
    import trio
    assert trio.run(handler, cid) == blob


def test_requires_host():
    with pytest.raises(ValueError):
        LibP2PBlobResolver(None)


# ---------------------------------------------------------------------------
# One host-level handler, shared by every resolver (and the BlobStore)
# ---------------------------------------------------------------------------


def _handler_host():
    """A fake host that, like AutonetHost, keeps ONE handler (last set wins)."""
    class Host:
        _trio_token = None
        known_capabilities: dict = {}
        handler = None
        n_set = 0

        def set_blob_handler(self, h):
            self.handler = h
            self.n_set += 1

    return Host()


def test_two_resolvers_share_one_handler_serving_world_and_carry():
    import trio

    host = _handler_host()
    chain_resolver = LibP2PBlobResolver(host)     # chain submission
    tool_resolver = LibP2PBlobResolver(host)      # tool_store fetches (built later)
    world = b'{"schema":1,"world":{}}'
    carry = b'{"carry":{},"epoch_id":"e1","schema":1}'
    world_cid = chain_resolver.put(world)
    carry_cid = chain_resolver.put(carry)
    LibP2PBlobResolver(host)                      # a third one, even later

    assert host.n_set == 1                        # never replaced
    assert trio.run(host.handler, world_cid) == world
    assert trio.run(host.handler, carry_cid) == carry
    # Every resolver on the host sees the shared store locally.
    assert tool_resolver.get(carry_cid) == carry
    assert tool_resolver.has(world_cid)


def test_blob_store_registers_as_source_not_replacement(tmp_path):
    import trio

    from nodes.common.blob_store import BlobStore

    host = _handler_host()
    resolver = LibP2PBlobResolver(host)
    carry_cid = resolver.put(b"carry-bundle")
    store = BlobStore(data_dir=str(tmp_path / "blobs"), p2p_host=host)
    disk_cid = store.add_bytes(b"on-disk manifest")

    assert host.n_set == 1
    assert trio.run(host.handler, carry_cid) == b"carry-bundle"
    assert trio.run(host.handler, disk_cid) == b"on-disk manifest"
    assert trio.run(host.handler, cid_for_blob(b"nowhere")) is None
