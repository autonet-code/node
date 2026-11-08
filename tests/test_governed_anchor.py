"""Governed anchoring through a real OZ Governor + timelock (2026-10-16).

eth_tester deploys a test-only DAO stand-in (contracts/test/
AnchorTestGovernance.sol): a timestamp-clocked ERC20Votes REP token, a
TimelockController and a Governor (Settings + CountingSimple + Votes +
QuorumFraction + TimelockControl). Substrate's governor is the timelock.
``GovernedAnchorer.step`` is driven with time travel between steps:

  - two independent GovernedAnchorer instances derive the same proposal
    id (the identical-proposal property every daemon relies on);
  - the proposal walks pending -> active -> succeeded -> queued ->
    executed and Substrate.isAnchored(epoch) is true afterwards;
  - a second propose of the identical proposal is swallowed by step();
  - Substrate's own governor-only check still rejects a direct call.

Plus the dev-network direct mode (Substrate governor = the daemon's EOA).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

eth_tester = pytest.importorskip("eth_tester")

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.governed_anchor import (
    ACTIVE,
    EXECUTED,
    PENDING,
    QUEUED,
    SUCCEEDED,
    GovernedAnchorer,
    Voter,
    anchor_description,
)
from tests.substrate_chain_helpers import ZERO, account_key, deploy_substrate

ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "contracts"
SUBSTRATE = ROOT / "core" / "Substrate.sol" / "Substrate.json"
GOV = ROOT / "test" / "AnchorTestGovernance.sol"

VOTING_DELAY = 1        # seconds (timestamp clock)
VOTING_PERIOD = 60
TIMELOCK_DELAY = 10
REP = 1000 * 10 ** 18


def _artifact(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"missing artifact: {path} (run npx hardhat compile)")
    return json.loads(path.read_text(encoding="utf-8"))


def _deploy(w3, art, *args):
    f = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
    tx = f.constructor(*args).transact({"from": w3.eth.accounts[0], "gas": 15_000_000})
    r = w3.eth.wait_for_transaction_receipt(tx)
    assert r.status == 1
    return w3.eth.contract(address=r.contractAddress, abi=art["abi"])


def _tx(w3, fn, sender):
    r = w3.eth.wait_for_transaction_receipt(
        fn.transact({"from": sender, "gas": 2_000_000}))
    assert r.status == 1
    return r


def _travel(w3, seconds: int) -> None:
    # eth_tester mines the travel block one second short of the target.
    t = w3.provider.ethereum_tester
    t.time_travel(int(w3.eth.get_block("latest")["timestamp"]) + seconds + 1)


def _close(epoch_id: str) -> dict:
    payload = {
        "epoch_root": "5a" * 32, "agent_mint": {}, "total_mint": 0.0,
        "output_decimals": 10, "n_batches": 1, "n_events": 0,
    }
    return {"authoritative_payload": payload, "agent_mint": {},
            "epoch_id": epoch_id}


@pytest.fixture
def dao():
    sub_art = _artifact(SUBSTRATE)
    rep_art = _artifact(GOV / "TestRepVotes.json")
    tl_art = _artifact(GOV / "TestTimelock.json")
    gov_art = _artifact(GOV / "TestAnchorGovernor.json")

    w3 = Web3(EthereumTesterProvider())
    accts = w3.eth.accounts
    deployer, agent, daemon_a, daemon_b = accts[0], accts[1], accts[2], accts[3]

    rep = _deploy(w3, rep_art)
    _tx(w3, rep.functions.mint(agent, REP), deployer)
    _tx(w3, rep.functions.delegate(agent), agent)
    # A non-delegated holder: counts in supply, votes nowhere.
    _tx(w3, rep.functions.mint(accts[4], REP // 4), deployer)

    timelock = _deploy(w3, tl_art, TIMELOCK_DELAY, [], [], deployer)
    governor = _deploy(w3, gov_art, rep.address, timelock.address,
                       VOTING_DELAY, VOTING_PERIOD, 4)
    for role in ("PROPOSER_ROLE", "EXECUTOR_ROLE", "CANCELLER_ROLE"):
        rid = getattr(timelock.functions, role)().call()
        _tx(w3, timelock.functions.grantRole(rid, governor.address), deployer)

    sub_addr = deploy_substrate(w3, sub_art["abi"], sub_art["bytecode"],
                                deployer, governor=timelock.address)
    substrate = w3.eth.contract(address=sub_addr, abi=sub_art["abi"])
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=sub_addr),
        web3=w3, contract_abi=sub_art["abi"])
    _travel(w3, 2)   # the delegation checkpoint is now in the past

    def ga(daemon):
        return GovernedAnchorer(w3=w3, substrate=substrate,
                                governor_address=governor.address,
                                chain_id=w3.eth.chain_id,
                                daemon_key=account_key(w3, daemon))

    return {
        "w3": w3, "substrate": substrate, "governor": governor,
        "timelock": timelock, "anchorer": anchorer,
        "ga_a": ga(daemon_a), "ga_b": ga(daemon_b),
        "voters": [Voter(agent, account_key(w3, agent))],
        "deployer": deployer,
    }


def test_governed_anchor_walks_the_full_lifecycle(dao):
    w3, substrate = dao["w3"], dao["substrate"]
    ga_a, ga_b, voters = dao["ga_a"], dao["ga_b"], dao["voters"]
    assert not ga_a.direct and ga_a.available

    prepared = dao["anchorer"].prepare_anchor(_close("e-gov-1"))
    args = prepared.fields.as_args()
    p = ga_a.build("e-gov-1", args)
    p_b = ga_b.build("e-gov-1", args)
    # Every daemon derives the identical proposal.
    assert p.proposal_id == p_b.proposal_id
    assert p.calldatas == p_b.calldatas
    assert p.description == anchor_description("e-gov-1", args[1])
    assert p.description == "Anchor epoch e-gov-1 0x" + "5a" * 32
    assert ga_a.state(p) is None

    # Propose (voting delay not over yet).
    assert ga_a.step(p, voters, may_propose=True) == "pending"
    assert ga_a.state(p) == PENDING

    # A second daemon's identical propose reverts inside step(), which
    # swallows it and carries on with the existing proposal.
    real_state = ga_b.state
    calls = {"n": 0}

    def first_none(prop):
        calls["n"] += 1
        return None if calls["n"] == 1 else real_state(prop)

    ga_b.state = first_none
    assert ga_b.step(p_b, voters, may_propose=True) == "pending"
    assert p_b.tx_hashes == []
    ga_b.state = real_state

    # Voting opens: the agent votes For with its delegated REP.
    _travel(w3, VOTING_DELAY + 1)
    assert ga_a.state(p) == ACTIVE
    assert ga_a.step(p, voters) == "active"
    assert dao["governor"].functions.hasVoted(p.proposal_id, voters[0].address).call()
    # Stepping again does not double-vote (hasVoted guard).
    n_tx = len(p.tx_hashes)
    assert ga_a.step(p, voters) == "active"
    assert len(p.tx_hashes) == n_tx

    # Deadline passes: succeeded, then step queues it in the timelock.
    _travel(w3, VOTING_PERIOD + 1)
    assert ga_a.state(p) == SUCCEEDED
    assert ga_a.step(p, voters) == "queued"
    assert ga_a.state(p) == QUEUED
    assert not substrate.functions.isAnchored("e-gov-1").call()

    # Timelock delay passes: any daemon executes.
    _travel(w3, TIMELOCK_DELAY + 1)
    assert ga_b.step(p_b, voters) == "executed"
    assert ga_a.state(p) == EXECUTED
    assert substrate.functions.isAnchored("e-gov-1").call()
    assert substrate.functions.anchorCount().call() == 1
    # Further steps are harmless no-ops.
    assert ga_a.step(p, voters) == "executed"


def test_direct_anchor_call_still_rejected_under_timelock(dao):
    prepared = dao["anchorer"].prepare_anchor(_close("e-direct"))
    r = dao["anchorer"].submit_prepared(prepared, from_address=dao["deployer"])
    assert not r.success
    assert not dao["substrate"].functions.isAnchored("e-direct").call()
    assert not dao["ga_a"].direct


def test_direct_mode_when_daemon_key_is_the_governor():
    sub_art = _artifact(SUBSTRATE)
    w3 = Web3(EthereumTesterProvider())
    daemon = w3.eth.accounts[2]
    addr = deploy_substrate(w3, sub_art["abi"], sub_art["bytecode"],
                            w3.eth.accounts[0], governor=daemon)
    substrate = w3.eth.contract(address=addr, abi=sub_art["abi"])
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=addr),
        web3=w3, contract_abi=sub_art["abi"])
    ga = GovernedAnchorer(w3=w3, substrate=substrate, governor_address="",
                          chain_id=w3.eth.chain_id,
                          daemon_key=account_key(w3, daemon))
    assert ga.direct and ga.available
    other = GovernedAnchorer(w3=w3, substrate=substrate, governor_address="",
                             chain_id=w3.eth.chain_id,
                             daemon_key=account_key(w3, w3.eth.accounts[3]))
    assert not other.direct and not other.available

    prepared = anchorer.prepare_anchor(_close("e-dev"))
    tx = ga.anchor_direct(prepared.fields.as_args())
    assert tx
    assert substrate.functions.isAnchored("e-dev").call()
    assert ZERO != substrate.functions.governor().call()
