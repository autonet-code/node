"""Phase 10.4: two-daemon smoke test (no chain).

In-process two-daemon scenario validating the full Phase 10 wiring
end-to-end without exercising the actual ``python -m atn`` CLI:

  - Each daemon has its own ``LibraryService`` (separate state dir) and
    its own ``EventGossip``.
  - The two gossip instances share an ``InMemoryHub`` so signed
    batches propagate between them in-process (no real libp2p).
  - Each daemon publishes a tool on behalf of one registered agent
    (with a 0x address).
  - When both daemons close their epoch, each runs the
    ``FederatedCloseDriver``: gather buffered batches → canonical
    order → federated_epoch_close → pick_submitter.
  - Assertion: both daemons compute the same ``authoritative_payload``
    bit-identical and the same ``winner``.

The chain side is covered separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

from nodes.common.event_gossip import (
    EventGossip,
    InMemoryHub,
    InMemoryTransport,
    Keypair,
)
from nodes.common.federated_close_driver import FederatedCloseDriver
from nodes.common.library.tool_manifest import build_tool_manifest
from nodes.common.library_service import LibraryService


def _make_daemon(
    tmp_path: Path,
    name: str,
    hub: InMemoryHub,
    rpb: str = "rpb_smoke",
) -> Tuple[LibraryService, EventGossip, FederatedCloseDriver]:
    """Build a LibraryService + EventGossip + FederatedCloseDriver triple
    sharing the given hub."""
    svc = LibraryService(rpb_address=rpb, data_root=tmp_path / name)
    transport = InMemoryTransport(hub=hub, node_id=name)
    gossip = EventGossip(
        rpb_address=rpb,
        keypair=Keypair.generate(),
        transport=transport,
        library=svc,
    )
    driver = FederatedCloseDriver(gossip=gossip)
    return svc, gossip, driver


def _publish_tool(svc: LibraryService, agent_address: str, name: str) -> str:
    """Publish one tool authored by ``agent_address``."""
    manifest = build_tool_manifest(
        name=name, description=f"{name}: does a thing.",
        input_schema={"type": "object", "properties": {}},
        author=agent_address, trust_class="pinned", code_digest="c" * 64,
        created_ts=1780531200)
    return svc.submit_tool_manifest(manifest, agent_id=agent_address)["manifest_digest"]


def test_two_daemons_reach_identical_federated_payload(tmp_path: Path):
    """Two daemons in one process gossip events between each other,
    then each runs the federated-close driver. Both must produce the
    same authoritative_payload and pick the same submitter."""
    hub = InMemoryHub()

    svc_a, gossip_a, driver_a = _make_daemon(tmp_path, "alpha", hub)
    svc_b, gossip_b, driver_b = _make_daemon(tmp_path, "beta", hub)

    try:
        # Open one epoch on each daemon.
        epoch_a = svc_a.open_epoch()
        epoch_b = svc_b.open_epoch()
        # Use the same epoch_id on both so on-chain dedup keys match.
        # (Real production gets this from a shared scheduler / clock.)
        epoch_id = "smoke_epoch_1"

        # Each daemon's agent publishes one tool. The local-events
        # subscriber wired by EventGossip publishes the registration to
        # the shared hub, which delivers it to the *other* daemon's gossip
        # instance, which buffers it for the canonical order.
        # InMemoryHub delivers synchronously.
        dig_a = _publish_tool(svc_a, "0xAAA", "alpha_tool")
        dig_b = _publish_tool(svc_b, "0xBBB", "beta_tool")

        # Close epoch on both daemons (local close — produces a
        # local result with epoch_id).
        local_a = svc_a.close_epoch()
        local_b = svc_b.close_epoch()

        # Stamp the same epoch_id on each so the federated driver
        # produces matching keys (in production, the scheduler aligns
        # epoch_id across daemons; in this test we just stamp).
        local_a["epoch_id"] = epoch_id
        local_b["epoch_id"] = epoch_id

        # Run the federated driver on each daemon.
        fed_a = driver_a.run(local_a)
        fed_b = driver_b.run(local_b)

        assert fed_a is not None, "daemon A produced no federated result"
        assert fed_b is not None, "daemon B produced no federated result"

        # Bit-identical authoritative payload across daemons.
        payload_a = fed_a.close_result["authoritative_payload"]
        payload_b = fed_b.close_result["authoritative_payload"]
        assert payload_a == payload_b, (
            "authoritative_payload diverged:\n"
            f"  A: {payload_a}\n  B: {payload_b}"
        )

        # Same winner agreed across daemons.
        assert fed_a.winner == fed_b.winner

        # Exactly one of the two daemons sees itself as winner.
        assert fed_a.is_winner != fed_b.is_winner, (
            "both daemons claim to be the winner — pick_submitter "
            "disagreement"
        )

        # Both daemons saw the same batches in canonical order (one per
        # published tool) and carry both registrations forward.
        assert fed_a.n_batches == fed_b.n_batches >= 2
        regs = fed_a.close_result["tool_registrations"]
        assert set(regs) == {dig_a, dig_b}
        assert regs == fed_b.close_result["tool_registrations"]
        assert fed_a.carry_cid == fed_b.carry_cid
    finally:
        svc_a.shutdown()
        svc_b.shutdown()


def test_isolated_daemons_dont_cross_contaminate(tmp_path: Path):
    """Sanity: two daemons on *different* RPBs (different gossip
    topics) do NOT see each other's events even on the same hub."""
    hub = InMemoryHub()

    svc_a, gossip_a, driver_a = _make_daemon(tmp_path, "alpha", hub, rpb="rpb_alpha")
    svc_b, gossip_b, driver_b = _make_daemon(tmp_path, "beta", hub, rpb="rpb_beta")

    try:
        svc_a.open_epoch()
        svc_b.open_epoch()
        _publish_tool(svc_a, "0xAAA", "alpha_tool")
        _publish_tool(svc_b, "0xBBB", "beta_tool")

        local_a = svc_a.close_epoch()
        local_b = svc_b.close_epoch()
        local_a["epoch_id"] = "e_iso_a"
        local_b["epoch_id"] = "e_iso_b"

        fed_a = driver_a.run(local_a)
        fed_b = driver_b.run(local_b)

        # Each daemon sees only its own batch in canonical order
        # because the other's batches went to a different gossip
        # topic.
        assert fed_a is not None
        assert fed_b is not None
        assert fed_a.n_batches == 1
        assert fed_b.n_batches == 1
        assert (fed_a.close_result["authoritative_payload"]
                != fed_b.close_result["authoritative_payload"])
    finally:
        svc_a.shutdown()
        svc_b.shutdown()
