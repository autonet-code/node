"""Phase 10.4: two-daemon smoke test (no chain).

In-process two-daemon scenario validating the full Phase 10 wiring
end-to-end without exercising the actual ``python -m atn`` CLI:

  - Each daemon has its own ``WorldService`` (separate persistence
    dir) and its own ``EventGossip``.
  - The two gossip instances share an ``InMemoryHub`` so signed
    batches propagate between them in-process (no real libp2p).
  - Each daemon submits observations on behalf of one registered
    agent (with a 0x address). 10.1's per-agent attribution makes
    sure mint accrues to the address, not to "daemon".
  - When both daemons close their epoch, each runs the
    ``FederatedCloseDriver``: gather buffered batches → canonical
    order → federated_epoch_close → pick_submitter.
  - Assertion: both daemons compute the same ``authoritative_payload``
    bit-identical and the same ``winner``.

This is what "two daemons grow a substrate together" means at the
pure-substrate layer — the chain side (10.3b/c/d) is separately
covered.
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
from nodes.common.world_service import WorldService


def _make_daemon(
    tmp_path: Path,
    name: str,
    hub: InMemoryHub,
    rpb: str = "rpb_smoke",
    embedding_dim: int = 32,
) -> Tuple[WorldService, EventGossip, FederatedCloseDriver]:
    """Build a WorldService + EventGossip + FederatedCloseDriver
    triple sharing the given hub.

    Pinned to the "equilibrated" kernel: this smoke submits
    observation-only events, which mint via geometric propagation +
    derived-sprout capture. Ledger mode (the post-phase8 default) scores
    net_score over causal events only, so observations alone don't move
    score — that path is exercised by tests/test_ledger_pricing.py.
    """
    svc = WorldService(
        rpb_address=rpb,
        data_root=tmp_path / name,
        embedding_dim=embedding_dim,
        pricing="equilibrated",
    )
    transport = InMemoryTransport(hub=hub, node_id=name)
    gossip = EventGossip(
        rpb_address=rpb,
        keypair=Keypair.generate(),
        transport=transport,
        world_service=svc,
    )
    driver = FederatedCloseDriver(
        gossip=gossip, embedding_dim=embedding_dim, pricing="equilibrated",
    )
    return svc, gossip, driver


def _submit_obs(svc: WorldService, agent_address: str, label: str, charter_axis: int):
    """Submit one observation from an agent biased toward one charter
    axis. Coords are 6-D (post Phase 9 charter expansion) + zeros for
    the embedding tail."""
    from world_model.generalized import Observation
    coords = [0.0] * 6
    coords[charter_axis] = 0.7
    # Embedding tail
    coords += [0.05 * i for i in range(svc.embedding_dim)]
    obs = Observation(id=f"obs_{label}", coords=tuple(coords), label=label)
    svc.submit_observation(obs, agent_id=agent_address)


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

        # Each daemon's agent submits one observation. The local-events
        # subscriber wired by EventGossip publishes those events to the
        # shared hub, which delivers them to the *other* daemon's
        # gossip instance, which submits them to the other daemon's
        # WorldService and buffers them for the canonical order.
        _submit_obs(svc_a, "0xAAA", "alpha_obs_1", charter_axis=0)
        _submit_obs(svc_b, "0xBBB", "beta_obs_1", charter_axis=2)

        # Allow a brief moment for InMemoryHub delivery (synchronous
        # but called from the WorldService's local-events thread; we
        # just need both events visible to both daemons).
        # InMemoryHub delivers synchronously, so by this point both
        # daemons should have buffered both batches.

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

        # Both daemons saw the same number of batches in canonical order
        # (one batch per daemon's submitted observation).
        assert fed_a.n_batches == fed_b.n_batches >= 2

        # Per-agent mint should attribute to both 0x addresses, not
        # to "default" / "daemon".
        agent_mint = payload_a.get("agent_mint", {})
        keys = set(agent_mint.keys())
        assert "0xAAA" in keys, f"missing 0xAAA mint: {keys}"
        assert "0xBBB" in keys, f"missing 0xBBB mint: {keys}"
        assert "default" not in keys
        assert "daemon" not in keys
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
        _submit_obs(svc_a, "0xAAA", "alpha_obs", charter_axis=0)
        _submit_obs(svc_b, "0xBBB", "beta_obs", charter_axis=1)

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
        agents_a = set(fed_a.close_result["authoritative_payload"]["agent_mint"].keys())
        agents_b = set(fed_b.close_result["authoritative_payload"]["agent_mint"].keys())
        assert "0xAAA" in agents_a
        assert "0xBBB" not in agents_a
        assert "0xBBB" in agents_b
        assert "0xAAA" not in agents_b
    finally:
        svc_a.shutdown()
        svc_b.shutdown()
