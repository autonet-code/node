"""Chain submission driver glue tests (governed anchors, 2026-10-16).

The pieces this driver depends on (EpochAnchorer, GovernedAnchorer,
AuthoritativeChainSubmitter) have their own eth_tester coverage in
test_phase5_5_anchor.py, test_governed_anchor.py and
test_phase5_6_authoritative_submission.py. This file tests the glue only:

  - Direct mode (Substrate's governor is this daemon's key): the winner
    anchors, then every daemon submits its agents' mints; non-winners do
    not anchor.
  - Governed mode: the close builds the DAO proposal, steps it once
    (winner may propose, non-winners may not) and parks it; ``tick()``
    keeps stepping and submits the agents' mints once the epoch is
    anchored; terminal proposals are dropped.
  - Per-agent submission walks the agent list and caches one submitter
    per address; blobs are published to the resolver.
  - Disabled mode, no-authority mode and stale snapshots are no-ops.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from nodes.common.blob_resolver import InMemoryBlobResolver
from nodes.common.chain_submission_driver import (
    AgentChainIdentity,
    ChainSubmissionConfig,
    ChainSubmissionDriver,
)
from nodes.common.epoch_anchorer import AnchorFields, AnchorResult, PreparedAnchor
from nodes.common.federated_close_driver import FederatedCloseResult
from nodes.common.governed_anchor import AnchorProposal

SUBMIT = ("nodes.common.chain_submission_driver."
          "AuthoritativeChainSubmitter.submit_for_epoch")


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


def _mock_anchorer(anchor_count=None):
    anchorer = MagicMock()
    anchorer.is_anchored.return_value = False
    if anchor_count is not None:
        anchorer.anchor_count.return_value = anchor_count

    def _prepare(close_result):
        fields = AnchorFields("e1", b"\x01" * 32, b"\x00" * 32, b"\x00" * 32,
                              "cid_xyz", b"\x02" * 32, b"\x03" * 32)
        return PreparedAnchor(fields=fields, result=AnchorResult(
            success=False, epoch_id="e1", agent_mint_cid="cid_xyz",
            agent_mint_blob=b"blob-bytes-here"))

    def _submit(prepared, **kw):
        r = prepared.result
        r.success = True
        r.tx_hash = "0xtx"
        return r

    anchorer.prepare_anchor.side_effect = _prepare
    anchorer.submit_prepared.side_effect = _submit
    return anchorer


def _mock_governed(*, direct: bool, available: bool = True, states=("active",)):
    """GovernedAnchorer stand-in. ``states`` are returned by successive
    step() calls (the last one repeats)."""
    ga = MagicMock()
    ga.direct = direct
    ga.available = direct or available

    def _build(epoch_id, args):
        return AnchorProposal(epoch_id=epoch_id, targets=["0x" + "1" * 40],
                              values=[0], calldatas=[b"\x00"],
                              description=f"Anchor epoch {epoch_id} 0x01",
                              description_hash=b"\x00" * 32, proposal_id=42)

    seq = list(states)

    def _step(p, voters, *, may_propose=True):
        st = seq.pop(0) if len(seq) > 1 else seq[0]
        p.last_state = st
        return st

    ga.build.side_effect = _build
    ga.step.side_effect = _step
    return ga


def _make_driver(*, enabled: bool = True,
                 agents: List[AgentChainIdentity] = None,
                 governed=None):
    cfg = ChainSubmissionConfig(
        substrate_address="0x" + "1" * 40 if enabled else "",
        rpc_url="http://localhost:8545",
        chain_id=31337,
        daemon_private_key="0x" + "2" * 64 if enabled else "",
    )
    if governed is None and enabled:
        governed = _mock_governed(direct=True)
    return ChainSubmissionDriver(
        config=cfg,
        agent_chain_resolver=lambda: list(agents or []),
        blob_resolver=InMemoryBlobResolver(),
        anchorer=_mock_anchorer() if enabled else None,
        governed=governed,
        tick_interval=3600.0,   # the background ticker never fires in tests
    )


@pytest.fixture
def fake_submit():
    r = MagicMock(success=True, tx_hash="0xagent_tx", raw_mint=1.0, error="")
    with patch(SUBMIT, return_value=r) as m:
        yield m


AGENTS = [
    AgentChainIdentity(address="0xAAA", private_key="0x" + "a" * 64),
    AgentChainIdentity(address="0xBBB", private_key="0x" + "b" * 64),
]


# ---------------------------------------------------------------------------
# Disabled / direct mode
# ---------------------------------------------------------------------------


def test_disabled_mode_is_no_op():
    driver = _make_driver(enabled=False)
    assert driver.enabled is False
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert out["anchored"] is False
    assert out["agent_submissions"] == []


def test_direct_winner_anchors_writes_blob_then_submits_agents(fake_submit):
    driver = _make_driver(agents=AGENTS)
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert out["anchored"] is True
    assert out["anchor_tx"] == "0xtx"
    assert driver.blob_resolver.has("cid_xyz") or len(driver.blob_resolver) > 0
    assert {r["claimant"] for r in out["agent_submissions"]} == {"0xAAA", "0xBBB"}
    driver._governed.build.assert_not_called()


def test_direct_non_winner_skips_anchor_but_submits(fake_submit):
    driver = _make_driver(agents=AGENTS)
    out = driver.handle_federated_close(_fed("e1", is_winner=False))
    assert out["anchored"] is False
    driver._anchorer.submit_prepared.assert_not_called()
    assert len(out["agent_submissions"]) == 2


def test_per_agent_submitter_cache(fake_submit):
    driver = _make_driver(agents=AGENTS)
    out1 = driver.handle_federated_close(_fed("e1", is_winner=True))
    out2 = driver.handle_federated_close(_fed("e2", is_winner=False))
    assert len(out1["agent_submissions"]) == 2
    assert len(out2["agent_submissions"]) == 2
    # One submitter per claimant, keyed lowercase.
    assert set(driver._submitter_cache.keys()) == {"0xaaa", "0xbbb"}


def test_agent_resolver_filtering(fake_submit):
    """An agent with no address is skipped. An agent with no private key
    is still claimed for: the operator key signs claims (Decision
    2026-10-27), the agent's own key is not needed."""
    agents = [
        AgentChainIdentity(address="0xAAA", private_key="0x" + "a" * 64),
        AgentChainIdentity(address="", private_key="0x" + "b" * 64),
        AgentChainIdentity(address="0xCCC", private_key=""),
    ]
    driver = _make_driver(agents=agents)
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert [r["claimant"] for r in out["agent_submissions"]] == ["0xAAA", "0xCCC"]


def test_owner_wallet_claimed_and_operator_signs(fake_submit):
    """The configured owner wallet is claimed for first (deduped against
    the agents), and every submitter is signed by the operator key, not
    the claimant's."""
    from eth_account import Account
    driver = _make_driver(agents=AGENTS)
    driver.config.owner_wallet = "0xDDD"
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert [r["claimant"] for r in out["agent_submissions"]] == [
        "0xDDD", "0xAAA", "0xBBB"]
    op_key = driver.config.daemon_private_key
    op_addr = Account.from_key(op_key).address
    for claimant, sub in driver._submitter_cache.items():
        assert sub.claimant_id == claimant
        assert sub.sender_address == op_addr
        assert sub.config.private_key == op_key


def test_no_operator_key_skips_claims(fake_submit):
    driver = _make_driver(agents=AGENTS)
    driver.config.daemon_private_key = ""
    assert driver._submit_agents("e1") == []
    fake_submit.assert_not_called()


def test_stale_snapshot_is_never_anchored():
    driver = _make_driver()
    driver._anchorer = _mock_anchorer(anchor_count=5)
    fed = _fed("e1", is_winner=True)
    fed.snapshot_anchor_count = 4
    out = driver.handle_federated_close(fed)
    assert "stale snapshot" in out["anchor_skipped"]
    assert out["anchored"] is False
    driver._anchorer.prepare_anchor.assert_not_called()


def test_already_anchored_skips_prepare(fake_submit):
    driver = _make_driver(agents=AGENTS)
    driver._anchorer.is_anchored.return_value = True
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert out["anchor_skipped"] == "already anchored"
    driver._anchorer.prepare_anchor.assert_not_called()
    assert len(out["agent_submissions"]) == 2


def test_no_authority_reports_error():
    ga = _mock_governed(direct=False, available=False)
    driver = _make_driver(governed=ga)
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert out["anchored"] is False
    assert "no DAO governor" in out["anchor_error"]
    driver._anchorer.submit_prepared.assert_not_called()


# ---------------------------------------------------------------------------
# Governed mode
# ---------------------------------------------------------------------------


def test_governed_winner_parks_proposal_and_tick_submits_once_anchored(fake_submit):
    ga = _mock_governed(direct=False, states=("active",))
    driver = _make_driver(agents=AGENTS, governed=ga)
    try:
        out = driver.handle_federated_close(_fed("e1", is_winner=True))
        assert out["anchored"] is False
        assert out["anchor_state"] == "active"
        assert out["anchor_proposal"] == hex(42)
        assert out["agent_submissions"] == []
        assert ga.step.call_args.kwargs["may_propose"] is True
        assert driver.pending_proposals == {"e1": "active"}
        driver._anchorer.submit_prepared.assert_not_called()
        fake_submit.assert_not_called()

        # Not anchored yet: tick steps again, still parked.
        driver.tick()
        assert ga.step.call_count == 2
        assert driver.pending_proposals == {"e1": "active"}
        fake_submit.assert_not_called()

        # The timelock executed it: the next tick submits the mints.
        driver._anchorer.is_anchored.return_value = True
        driver.tick()
        assert fake_submit.call_count == 2
        assert driver.pending_proposals == {}
        assert ga.step.call_count == 2   # no step once anchored
    finally:
        driver.stop()


def test_governed_non_winner_does_not_propose_on_close():
    ga = _mock_governed(direct=False, states=("nonexistent",))
    driver = _make_driver(governed=ga)
    try:
        out = driver.handle_federated_close(_fed("e1", is_winner=False))
        assert ga.step.call_args.kwargs["may_propose"] is False
        assert out["anchor_state"] == "nonexistent"
        # Within one tick interval of the close it still may not propose.
        driver.tick()
        assert ga.step.call_args.kwargs["may_propose"] is False
        # After a tick interval, a non-winner proposes too.
        driver._pending["e1"].created -= 2 * driver.tick_interval
        driver.tick()
        assert ga.step.call_args.kwargs["may_propose"] is True
    finally:
        driver.stop()


def test_governed_executed_in_one_step_submits_immediately(fake_submit):
    ga = _mock_governed(direct=False, states=("executed",))
    driver = _make_driver(agents=AGENTS, governed=ga)
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert out["anchored"] is True
    assert len(out["agent_submissions"]) == 2
    assert driver.pending_proposals == {}


def test_governed_tick_executed_state_submits(fake_submit):
    ga = _mock_governed(direct=False, states=("active", "executed"))
    driver = _make_driver(agents=AGENTS, governed=ga)
    try:
        driver.handle_federated_close(_fed("e1", is_winner=True))
        driver.tick()
        assert fake_submit.call_count == 2
        assert driver.pending_proposals == {}
    finally:
        driver.stop()


def test_governed_defeated_proposal_is_dropped(fake_submit):
    ga = _mock_governed(direct=False, states=("active", "defeated"))
    driver = _make_driver(agents=AGENTS, governed=ga)
    try:
        driver.handle_federated_close(_fed("e1", is_winner=True))
        driver.tick()
        assert driver.pending_proposals == {}
        fake_submit.assert_not_called()
    finally:
        driver.stop()


def test_governed_build_failure_reports_error():
    ga = _mock_governed(direct=False)
    ga.build.side_effect = RuntimeError("rpc down")
    driver = _make_driver(governed=ga)
    out = driver.handle_federated_close(_fed("e1", is_winner=True))
    assert "proposal build failed" in out["anchor_error"]
    assert driver.pending_proposals == {}


def test_operator_claims_rep_after_atn_for_households_with_earnings():
    """REP follows the ATN (Decision 2026-10-27): after the claims, the
    operator calls AutonetRepClaim.claim for each household with REP
    claimable, and skips the ones with none."""
    from nodes.common.chain_submission_driver import Voter
    d = _make_driver()
    d.config.rep_claim_address = "0x" + "3" * 40
    claimable = {"0x" + "a" * 40: 5, "0x" + "b" * 40: 0}
    claimed = []
    rc = MagicMock()
    rc.functions.claimable.side_effect = lambda who: MagicMock(
        call=lambda: claimable[who.lower()])

    def _claim(who):
        claimed.append(who.lower())
        return MagicMock(build_transaction=lambda tx: {
            **tx, "to": "0x" + "3" * 40, "value": 0, "gas": 100_000,
            "gasPrice": 1, "data": "0x"})
    rc.functions.claim.side_effect = _claim
    w3 = MagicMock()
    w3.to_checksum_address.side_effect = lambda a: a
    w3.eth.contract.return_value = rc
    w3.eth.get_transaction_count.return_value = 0
    w3.eth.wait_for_transaction_receipt.return_value = MagicMock(status=1)
    d._submitter_web3 = w3
    d._claim_rep(list(claimable), Voter(address="0x" + "c" * 40,
                                        private_key="0x" + "4" * 64))
    assert claimed == ["0x" + "a" * 40]
    assert w3.eth.send_raw_transaction.call_count == 1
