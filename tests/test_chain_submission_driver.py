"""Phase 10.3b/c: chain submission driver glue tests.

The pieces this driver depends on (EpochAnchorer, AuthoritativeChainSubmitter)
have their own end-to-end eth_tester coverage in test_phase5_5_anchor.py
and test_phase5_6_authoritative_submission.py. This file tests the
glue layer only:

  - Winner daemons call anchor; non-winners skip it.
  - Per-agent submission walks the agent list and constructs one
    submitter per address (cached).
  - Blob from a successful anchor is written to the resolver so
    per-agent submitters can read it.
  - Disabled mode (no substrate_address / no key) is a no-op,
    not a crash.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

from nodes.common.blob_resolver import InMemoryBlobResolver
from nodes.common.chain_submission_driver import (
    AgentChainIdentity,
    ChainSubmissionConfig,
    ChainSubmissionDriver,
)
from nodes.common.epoch_anchorer import AnchorResult
from nodes.common.federated_close_driver import FederatedCloseResult


def _fed(epoch_id: str, is_winner: bool, payload=None) -> FederatedCloseResult:
    close_result = {
        "epoch_id": epoch_id,
        "authoritative_payload": payload or {
            "agent_mint": {"0xAAA": 1.0},
            "epoch_root": "abc123",
        },
        "agent_mint": {"0xAAA": 1.0},
    }
    return FederatedCloseResult(
        epoch_id=epoch_id,
        close_result=close_result,
        senders=[b"x" * 32, b"y" * 32],
        winner=b"x" * 32,
        is_winner=is_winner,
        n_batches=2,
    )


def _make_driver(
    *,
    enabled: bool = True,
    agents: List[AgentChainIdentity] = None,
):
    """Build a ChainSubmissionDriver with a mocked anchorer so we
    don't touch a real chain."""
    cfg = ChainSubmissionConfig(
        substrate_address="0x" + "1" * 40 if enabled else "",
        rpc_url="http://localhost:8545",
        chain_id=31337,
        daemon_private_key="0x" + "2" * 64 if enabled else "",
    )
    anchorer = MagicMock() if enabled else None
    if anchorer is not None:
        anchorer.anchor_close_result.return_value = AnchorResult(
            success=True,
            tx_hash="0xtx",
            epoch_id="e1",
            epoch_root_hex="abc123",
            agent_mint_cid="cid_xyz",
            agent_mint_blob=b"blob-bytes-here",
        )
    return ChainSubmissionDriver(
        config=cfg,
        agent_chain_resolver=lambda: list(agents or []),
        blob_resolver=InMemoryBlobResolver(),
        anchorer=anchorer,
    )


def test_disabled_mode_is_no_op():
    driver = _make_driver(enabled=False)
    assert driver.enabled is False
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert out["anchored"] is False
    assert out["agent_submissions"] == []


def test_winner_anchors_and_writes_blob():
    driver = _make_driver()
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert out["anchored"] is True
    assert out["anchor_tx"] == "0xtx"
    # Blob from anchor result should now be in the resolver.
    assert driver.blob_resolver.has("cid_xyz") or len(driver.blob_resolver) > 0


def test_non_winner_skips_anchor():
    driver = _make_driver()
    # Make sure anchorer is NOT called when not the winner.
    out = driver.handle_federated_close(_fed("e1", is_winner=False))
    assert out["anchored"] is False
    driver._anchorer.anchor_close_result.assert_not_called()


def test_per_agent_submitter_cache():
    """Two close events => same submitter instance reused per agent
    address."""
    agents = [
        AgentChainIdentity(address="0xAAA", private_key="0x" + "a" * 64),
        AgentChainIdentity(address="0xBBB", private_key="0x" + "b" * 64),
    ]
    driver = _make_driver(agents=agents)

    # Mock submit_for_epoch on each cached submitter so we don't
    # hit a real chain.
    fake_result = MagicMock()
    fake_result.success = True
    fake_result.tx_hash = "0xagent_tx"
    fake_result.raw_mint = 1.0
    fake_result.error = ""

    with patch(
        "nodes.common.chain_submission_driver.AuthoritativeChainSubmitter.submit_for_epoch",
        return_value=fake_result,
    ):
        out1 = driver.handle_federated_close(_fed("e1", is_winner=True))
        out2 = driver.handle_federated_close(_fed("e2", is_winner=False))

    assert len(out1["agent_submissions"]) == 2
    assert len(out2["agent_submissions"]) == 2
    # Both calls returned for both agents.
    addrs = {r["agent_address"] for r in out1["agent_submissions"]}
    assert addrs == {"0xAAA", "0xBBB"}

    # Cache should hold one submitter per agent address.
    assert set(driver._submitter_cache.keys()) == {"0xAAA", "0xBBB"}


def test_agent_resolver_filtering():
    """Agents missing address or private_key are silently skipped."""
    agents = [
        AgentChainIdentity(address="0xAAA", private_key="0x" + "a" * 64),
        AgentChainIdentity(address="", private_key="0x" + "b" * 64),  # no addr
        AgentChainIdentity(address="0xCCC", private_key=""),           # no key
    ]
    driver = _make_driver(agents=agents)

    fake_result = MagicMock()
    fake_result.success = True
    fake_result.tx_hash = "0xagent_tx"
    fake_result.raw_mint = 0.5
    fake_result.error = ""

    with patch(
        "nodes.common.chain_submission_driver.AuthoritativeChainSubmitter.submit_for_epoch",
        return_value=fake_result,
    ):
        out = driver.handle_federated_close(_fed("e1", is_winner=True))

    assert len(out["agent_submissions"]) == 1
    assert out["agent_submissions"][0]["agent_address"] == "0xAAA"
