"""The daemon side of anchor elections (nodes/common/anchor_election.py)
against the real stack: the DAO's HomebaseDAO with options counting
(trustless-contracts 6d1a5c6+, its compiled artifact), a timestamp REP
token and timelock (contracts/test/AnchorTestGovernance.sol), Substrate
and AnchorElection.

Two daemons with different agents follow the same epoch: both derive the
same proposal id, one proposes, drawn agents register options, every agent
votes for the option matching its own daemon's close, the winner anchors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

eth_tester = pytest.importorskip("eth_tester")

from eth_utils import keccak, to_checksum_address
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from nodes.common.anchor_election import ElectionAnchorer, epoch_number_of
from nodes.common.governed_anchor import Voter
from tests.substrate_chain_helpers import (
    account_key,
    deploy_substrate,
    register_bound,
)

ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "contracts"
GOV = ROOT / "test" / "AnchorTestGovernance.sol"
HOMEBASE = Path("C:/code/dao/trustless-contracts/artifacts/contracts/Dao.sol/HomebaseDAO.json")
REP = 1000 * 10 ** 18


def _artifact(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _deploy(w3, art, *args):
    f = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
    tx = f.constructor(*args).transact({"from": w3.eth.accounts[0], "gas": 15_000_000})
    r = w3.eth.wait_for_transaction_receipt(tx)
    assert r.status == 1
    return w3.eth.contract(address=r.contractAddress, abi=art["abi"])


def _tx(w3, fn, sender):
    r = w3.eth.wait_for_transaction_receipt(
        fn.transact({"from": sender, "gas": 3_000_000}))
    assert r.status == 1
    return r


def _travel(w3, seconds: int) -> None:
    t = w3.provider.ethereum_tester
    t.time_travel(int(w3.eth.get_block("latest")["timestamp"]) + seconds + 1)


def _next_address(w3, sender: str, ahead: int = 0) -> str:
    import rlp
    nonce = w3.eth.get_transaction_count(sender) + ahead
    return to_checksum_address(keccak(rlp.encode(
        [bytes.fromhex(sender[2:]), nonce]))[12:])


@pytest.fixture
def stack():
    hb = _artifact(HOMEBASE)
    names = [i["name"] for i in next(
        f for f in hb["abi"] if f.get("type") == "constructor")["inputs"]]
    if "_launcher" not in names:
        pytest.skip("HomebaseDAO artifact predates the options launcher")
    rep_art = _artifact(GOV / "TestRepVotes.json")
    tl_art = _artifact(GOV / "TestTimelock.json")
    sub_art = _artifact(ROOT / "core" / "Substrate.sol" / "Substrate.json")
    el_art = _artifact(ROOT / "core" / "AnchorElection.sol" / "AnchorElection.json")

    w3 = Web3(EthereumTesterProvider())
    acc = w3.eth.accounts
    deployer = acc[0]
    agents = acc[1:5]
    rep = _deploy(w3, rep_art)
    for a in agents:
        _tx(w3, rep.functions.mint(a, REP), deployer)
        _tx(w3, rep.functions.delegate(a), a)
    timelock = _deploy(w3, tl_art, 0, [], ["0x" + "00" * 20], deployer)
    # 1 minute voting delay, 3 minutes voting, threshold 0, quorum 4%.
    governor = _deploy(w3, hb, rep.address, timelock.address, "Test DAO",
                       1, 3, 0, 4, deployer)

    # Substrate takes the election's address before it exists.
    election_addr = _next_address(w3, deployer, ahead=1)
    sub_addr = deploy_substrate(w3, sub_art["abi"], sub_art["bytecode"],
                                deployer, governor=timelock.address,
                                anchorer=election_addr)
    substrate = w3.eth.contract(address=sub_addr, abi=sub_art["abi"])
    election = _deploy(w3, el_art, sub_addr, governor.address, rep.address)
    assert election.address == election_addr
    assert substrate.functions.anchorer().call() == election_addr

    # The launcher approves the source, then hands the timelock over.
    _tx(w3, governor.functions.setOptionSource(election.address, True), deployer)
    for role in ("PROPOSER_ROLE", "CANCELLER_ROLE"):
        rid = getattr(timelock.functions, role)().call()
        _tx(w3, timelock.functions.grantRole(rid, governor.address), deployer)

    for i, a in enumerate(agents):
        assert register_bound(w3, substrate, a, f"peer-{i}".encode()).status == 1
    _travel(w3, 2)

    def daemon(agent_slice):
        voters = [Voter(a, account_key(w3, a)) for a in agent_slice]
        ea = ElectionAnchorer(
            w3=w3, substrate=substrate, election_address=election.address,
            governor_address=governor.address, chain_id=w3.eth.chain_id,
            daemon_key=voters[0].private_key, election_abi=el_art["abi"])
        return ea, voters

    return {"w3": w3, "substrate": substrate, "governor": governor,
            "election": election, "daemon_a": daemon(agents[:2]),
            "daemon_b": daemon(agents[2:])}


def _option(stack, tag: str) -> tuple:
    sub, w3 = stack["substrate"], stack["w3"]
    cutoff = max(int(w3.eth.get_block("latest")["timestamp"]),
                 int(sub.functions.latestCutoff().call()) + 1)
    return (f"epoch-1-{'0' * 16}-{cutoff}", keccak(text=f"root-{tag}"),
            bytes(sub.functions.latestEpochRoot().call()),
            bytes(sub.functions.latestAnchorHash().call()),
            f"cid-{tag}", keccak(text=f"payload-{tag}"), bytes(32), cutoff)


def _run(stack, elections, rounds=40):
    """Step every daemon's election, moving time forward between rounds."""
    w3 = stack["w3"]
    for _ in range(rounds):
        for (ea, voters), e in elections:
            ea.step(e, voters)
        if stack["substrate"].functions.anchorCount().call() > 0:
            return
        _travel(w3, 20)   # daemons tick every 15 s; the voting delay is 60 s


def test_two_daemons_agree_and_anchor(stack):
    option = _option(stack, "same")
    (ea_a, v_a), (ea_b, v_b) = stack["daemon_a"], stack["daemon_b"]
    e_a = ea_a.open(option[0], option)
    e_b = ea_b.open(option[0], option)
    assert e_a.epoch_number == e_b.epoch_number == epoch_number_of(option[0]) == 1
    assert ea_a.proposal_id(e_a) == ea_b.proposal_id(e_b)

    _run(stack, [((ea_a, v_a), e_a), ((ea_b, v_b), e_b)])

    sub = stack["substrate"]
    assert sub.functions.anchorCount().call() == 1
    anchored = sub.functions.getAnchor(0).call()
    assert anchored[0] == option[0] and anchored[13] == option[7]
    # Agreeing daemons registered one option between them.
    pid = ea_a.proposal_id(e_a)
    assert stack["election"].functions.optionCount(pid).call() == 1


def test_split_vote_resolves_to_one_anchor(stack):
    """Daemon A and daemon B (two agents and equal REP each) disagree: each
    votes for its own option, the tie goes to the lowest option hash, and
    exactly one anchor lands."""
    a_opt, b_opt = _option(stack, "a"), _option(stack, "b")
    (ea_a, v_a), (ea_b, v_b) = stack["daemon_a"], stack["daemon_b"]
    e_a, e_b = ea_a.open(a_opt[0], a_opt), ea_b.open(b_opt[0], b_opt)
    assert ea_a.proposal_id(e_a) == ea_b.proposal_id(e_b)
    _run(stack, [((ea_a, v_a), e_a), ((ea_b, v_b), e_b)])
    sub = stack["substrate"]
    assert sub.functions.anchorCount().call() == 1
    winner = min(e_a.option_hash, e_b.option_hash)
    got = bytes(stack["election"].functions.optionHashOf(
        tuple(sub.functions.getAnchor(0).call()[i] for i in
              (0, 1, 2, 3, 4, 5, 9, 13))).call())
    assert got == winner


def test_pipelined_elections_land_in_order(stack):
    """Epoch 2's election runs before epoch 1 is anchored: its option
    chains onto epoch 1's option (root + content-only anchor hash), its
    execution waits for epoch 1, then both land in order."""
    sub = stack["substrate"]
    (ea_a, v_a), (ea_b, v_b) = stack["daemon_a"], stack["daemon_b"]
    opt1 = _option(stack, "one")
    c2 = opt1[7] + 300
    opt2 = (f"epoch-2-{opt1[1].hex()[:16]}-{c2}", keccak(text="root-two"),
            bytes(opt1[1]), bytes(sub.functions.anchorHashOf(*opt1).call()),
            "cid-two", keccak(text="payload-two"), bytes(32), c2)
    elections = []
    for ea, voters in ((ea_a, v_a), (ea_b, v_b)):
        elections.append(((ea, voters), ea.open(opt1[0], opt1)))
        elections.append(((ea, voters), ea.open(opt2[0], opt2)))
    w3 = stack["w3"]
    for _ in range(60):
        for (ea, voters), e in elections:
            ea.step(e, voters)
        if sub.functions.anchorCount().call() >= 2:
            break
        _travel(w3, 20)
    assert sub.functions.anchorCount().call() == 2
    assert [sub.functions.getAnchor(i).call()[0] for i in (0, 1)] == [opt1[0], opt2[0]]
