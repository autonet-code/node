"""AnchorElection (Decision 2026-10-26, items 5-6 design) + the Substrate
changes it relies on: an explicit increasing cutoff, a content-only anchor
hash, fee windows between cutoffs, and the anchorer role.

The Governor is MockOptionGovernor (its own timelock; the test sets each
proposal's state and winner), standing in for HomebaseDAO with the options
counting module that trustless-contracts builds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

eth_tester = pytest.importorskip("eth_tester")

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from tests.substrate_chain_helpers import (
    anchor_direct,
    deploy_substrate,
    register_bound,
)

ART = Path("C:/code/autonet/artifacts/contracts")
PENDING, ACTIVE = 0, 1


def _art(rel: str):
    p = ART / rel
    if not p.exists():
        pytest.skip(f"missing artifact: {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["abi"], d["bytecode"]


def _deploy(w3, rel, *args):
    abi, bc = _art(rel)
    tx = w3.eth.contract(abi=abi, bytecode=bc).constructor(*args).transact(
        {"from": w3.eth.accounts[0], "gas": 8_000_000})
    r = w3.eth.wait_for_transaction_receipt(tx)
    assert r.status == 1
    return w3.eth.contract(address=r.contractAddress, abi=abi)


def _send(w3, fn, sender, gas=3_000_000):
    tx = fn.transact({"from": sender, "gas": gas})
    return w3.eth.wait_for_transaction_receipt(tx)


def _reverts(w3, fn, sender) -> bool:
    try:
        return _send(w3, fn, sender).status != 1
    except Exception:
        return True


@pytest.fixture
def env():
    w3 = Web3(EthereumTesterProvider())
    acc = w3.eth.accounts
    sub_abi, sub_bc = _art("core/Substrate.sol/Substrate.json")
    sub = w3.eth.contract(
        address=deploy_substrate(w3, sub_abi, sub_bc, acc[0]), abi=sub_abi)
    rep = _deploy(w3, "test/MockRepToken.sol/MockRepToken.json")
    gov = _deploy(w3, "test/MockOptionGovernor.sol/MockOptionGovernor.json")
    el = _deploy(w3, "core/AnchorElection.sol/AnchorElection.json",
                 sub.address, gov.address, rep.address)
    assert _send(w3, sub.functions.setAnchorer(el.address), acc[0]).status == 1
    agents = acc[1:6]
    for i, a in enumerate(agents):
        assert register_bound(w3, sub, a, f"peer-{i}".encode()).status == 1
    _send(w3, gov.functions.setVotingDelay(10), acc[0])
    return {"w3": w3, "acc": acc, "sub": sub, "rep": rep, "gov": gov,
            "el": el, "agents": agents}


def _ts(w3) -> int:
    return int(w3.eth.get_block("latest")["timestamp"])


def _open(env, epoch, attempt=0, state=PENDING):
    """Create the election's proposal in the mock, created at the head
    block (so REP set before it counts)."""
    w3, gov, el = env["w3"], env["gov"], env["el"]
    pid = el.functions.proposalIdOf(epoch, attempt).call()
    _send(w3, gov.functions.setProposal(pid, state, _ts(w3) + 10),
          env["acc"][0])
    return pid


def _option(env, tag: str, cutoff=None):
    sub, w3 = env["sub"], env["w3"]
    return (
        f"epoch-{tag}",
        Web3.keccak(text=f"root-{tag}"),
        bytes(sub.functions.latestEpochRoot().call()),
        bytes(sub.functions.latestAnchorHash().call()),
        f"cid-{tag}",
        Web3.keccak(text=f"payload-{tag}"),
        bytes(32),
        _ts(w3) if cutoff is None else cutoff,
    )


def _give_rep(env, account, amount):
    _send(env["w3"], env["rep"].functions.setVotes(account, amount),
          env["acc"][0])


def test_register_vote_and_anchor_the_winner(env):
    w3, el, sub, gov = env["w3"], env["el"], env["sub"], env["gov"]
    a0, a1, a2 = env["agents"][:3]
    _give_rep(env, a0, 1000)
    _give_rep(env, a1, 1000)
    pid = _open(env, 1)

    opt_a, opt_b = _option(env, "a"), _option(env, "b")
    assert _send(w3, el.functions.registerOption(1, 0, opt_a), a0).status == 1
    # One option per agent; identical content registers once.
    assert _reverts(w3, el.functions.registerOption(1, 0, opt_b), a0)
    assert _reverts(w3, el.functions.registerOption(1, 0, opt_a), a1)
    assert _send(w3, el.functions.registerOption(1, 0, opt_b), a1).status == 1
    # No REP: never drawn while the election is not open to all.
    assert _reverts(w3, el.functions.registerOption(1, 0, _option(env, "c")), a2)
    assert el.functions.optionCount(pid).call() == 2
    _, proposer = el.functions.getOption(pid, 1).call()
    assert proposer == a1

    # Voting started: the option set is frozen.
    _send(w3, gov.functions.setProposal(pid, ACTIVE, _ts(w3)), env["acc"][0])
    assert _reverts(w3, el.functions.registerOption(1, 0, _option(env, "d")), a2)

    # Only the timelock executes, and only a decided election.
    call = el.encode_abi("anchorWinner", args=[1, 0])
    assert _reverts(w3, el.functions.anchorWinner(1, 0), a0)
    assert _reverts(w3, gov.functions.exec(el.address, call), env["acc"][0])
    _send(w3, gov.functions.setWinner(pid, True, 1), env["acc"][0])
    assert _send(w3, gov.functions.exec(el.address, call), env["acc"][0]).status == 1

    assert sub.functions.anchorCount().call() == 1
    anchored = sub.functions.getAnchor(0).call()
    assert anchored[0] == "epoch-b" and anchored[13] == opt_b[7]
    # The anchor hash is content-only, so it was computable in advance.
    assert bytes(sub.functions.latestAnchorHash().call()) == bytes(
        sub.functions.anchorHashOf(*opt_b).call())
    # Epoch 1 is done; executing it again is the wrong epoch.
    assert _reverts(w3, gov.functions.exec(el.address, call), env["acc"][0])


def test_next_election_waits_for_the_previous_anchor(env):
    w3, el, gov = env["w3"], env["el"], env["gov"]
    a0 = env["agents"][0]
    _give_rep(env, a0, 1000)
    pid2 = _open(env, 2)
    assert _send(w3, el.functions.registerOption(2, 0, _option(env, "x")),
                 a0).status == 1
    _send(w3, gov.functions.setWinner(pid2, True, 0), env["acc"][0])
    call = el.encode_abi("anchorWinner", args=[2, 0])
    # Epoch 1 is not anchored yet: epoch 2 cannot land (stays queued).
    assert _reverts(w3, gov.functions.exec(el.address, call), env["acc"][0])


def test_open_after_attempts_lets_any_live_agent_register(env):
    w3, el = env["w3"], env["el"]
    a2, outsider = env["agents"][2], env["acc"][7]
    _open(env, 1, attempt=2)
    assert _send(w3, el.functions.registerOption(1, 2, _option(env, "o")),
                 a2).status == 1
    # Still agents only.
    assert _reverts(w3, el.functions.registerOption(1, 2, _option(env, "p")),
                    outsider)


def test_draw_size_tracks_proposers(env):
    w3, el = env["w3"], env["el"]
    for a in env["agents"]:
        _give_rep(env, a, 100)
    tp = _ts(w3) - 1
    counts = [sum(el.functions.isDrawn(e, 0, a, tp).call()
                  for a in env["agents"]) for e in range(1, 61)]
    mean = sum(counts) / len(counts)
    # 5 equal holders, 3 expected proposers: each drawn with p = 0.6.
    assert 2.4 <= mean <= 3.6, mean


def test_params_are_governed(env):
    w3, el, gov = env["w3"], env["el"], env["gov"]
    assert _reverts(w3, el.functions.setParams(5, 0, 8, 2), env["acc"][0])
    call = el.encode_abi("setParams", args=[5, 1000, 4, 3])
    assert _send(w3, gov.functions.exec(el.address, call), env["acc"][0]).status == 1
    assert el.functions.proposers().call() == 5
    assert el.functions.proposerFloorPpm().call() == 1000
    bad = el.encode_abi("setParams", args=[0, 0, 8, 2])
    assert _reverts(w3, gov.functions.exec(el.address, bad), env["acc"][0])


def test_fee_windows_between_cutoffs(env):
    """windowBurned counts fees burned between the previous cutoff and this
    one, so a later anchor of an earlier cutoff gets the same pool."""
    w3, sub, acc = env["w3"], env["sub"], env["acc"]
    payer, recip = acc[7], acc[6]
    _send(w3, sub.functions.mintFromVault(payer, 10 ** 9), acc[8])

    def pay(amount):
        _send(w3, sub.functions.payForService(recip, amount, bytes(32)), payer)
        return int(sub.functions.totalBurned().call())

    pay(10 ** 8)
    c1 = _ts(w3)
    burned_c1 = int(sub.functions.totalBurned().call())
    pay(10 ** 8)            # after c1: belongs to the next window
    assert anchor_direct(w3, sub, "w1", b"\x01" * 32, "c", b"\x02" * 32,
                         bytes(32), cutoff=c1).status == 1
    assert sub.functions.getAnchor(0).call()[10] == burned_c1
    assert int(sub.functions.burnedSinceAnchor().call()) > 0
    def refused(**kw) -> bool:
        try:
            return anchor_direct(w3, sub, "w2", b"\x03" * 32, "c",
                                 b"\x04" * 32, bytes(32), **kw).status != 1
        except Exception:
            return True

    # A cutoff not after the latest one is refused.
    assert refused(cutoff=c1)
    # Only the governor or the anchorer anchors.
    assert refused(sender=acc[3])
    assert sub.functions.anchorCount().call() == 1
