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
    """Open the epoch's election up to `attempt` through the contract (an
    earlier attempt is ended as Defeated first), then set the proposal's
    state and its snapshot at the head block (so REP set before it counts)."""
    w3, gov, el = env["w3"], env["gov"], env["el"]
    opener = env["agents"][0]
    while el.functions.attemptsOpened(epoch).call() <= attempt:
        opened = el.functions.attemptsOpened(epoch).call()
        if opened:
            prev = el.functions.proposalIdOf(epoch, opened - 1).call()
            _send(w3, gov.functions.setProposal(prev, 3, _ts(w3)), env["acc"][0])
        assert _send(w3, el.functions.open(epoch), opener).status == 1
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
    # Index 0 is reject, so two registered options count as three.
    assert el.functions.optionCount(pid).call() == 3
    _, proposer = el.functions.getOption(pid, 2).call()
    assert proposer == a1

    # Registration stays open while voting runs, but still needs REP.
    _send(w3, gov.functions.setProposal(pid, ACTIVE, _ts(w3)), env["acc"][0])
    assert _reverts(w3, el.functions.registerOption(1, 0, _option(env, "d")), a2)

    # Only the timelock executes, and only a decided election.
    call = el.encode_abi("anchorWinner", args=[1, 0])
    assert _reverts(w3, el.functions.anchorWinner(1, 0), a0)
    assert _reverts(w3, gov.functions.exec(el.address, call), env["acc"][0])
    _send(w3, gov.functions.setWinner(pid, True, 2), env["acc"][0])
    assert _send(w3, gov.functions.exec(el.address, call), env["acc"][0]).status == 1

    assert sub.functions.anchorCount().call() == 1
    anchored = sub.functions.getAnchor(0).call()
    assert anchored[0] == "epoch-b" and anchored[10] == opt_b[7]
    # The anchor hash is content-only, so it was computable in advance.
    assert bytes(sub.functions.latestAnchorHash().call()) == bytes(
        sub.functions.anchorHashOf(*opt_b).call())
    # Epoch 1 is done; executing it again is the wrong epoch.
    assert _reverts(w3, gov.functions.exec(el.address, call), env["acc"][0])


def test_only_the_anchoring_call_is_an_election_call(env):
    """The governor runs a proposal as an election only when its one action
    is the election's own call; anything else on this contract (setParams)
    is an ordinary For/Against proposal."""
    el = env["el"]
    assert el.functions.isElectionCall(el.functions.callOf(1, 0).call()).call()
    assert el.functions.isElectionCall(
        el.encode_abi("anchorWinner", args=[7, 3])).call()
    assert not el.functions.isElectionCall(
        el.encode_abi("setParams", args=[1, 1_000_000, 1, 2])).call()
    assert not el.functions.isElectionCall(b"").call()
    assert not el.functions.isElectionCall(
        bytes.fromhex(el.encode_abi("anchorWinner", args=[1, 0])[2:8])).call()
    # The selector alone is enough: the arguments are checked on execution.
    assert el.functions.isElectionCall(
        bytes.fromhex(el.encode_abi("anchorWinner", args=[1, 0])[2:10])).call()


def test_next_election_waits_for_the_previous_anchor(env):
    w3, el, gov = env["w3"], env["el"], env["gov"]
    a0 = env["agents"][0]
    _give_rep(env, a0, 1000)
    # Epoch 2 opens once epoch 1's election has a winner waiting to land.
    pid1 = _open(env, 1, state=4)
    _send(w3, gov.functions.setWinner(pid1, True, 1), env["acc"][0])
    pid2 = _open(env, 2)
    assert _send(w3, el.functions.registerOption(2, 0, _option(env, "x")),
                 a0).status == 1
    _send(w3, gov.functions.setWinner(pid2, True, 0), env["acc"][0])
    call = el.encode_abi("anchorWinner", args=[2, 0])
    # Epoch 1 is not anchored yet: epoch 2 cannot land (stays queued).
    assert _reverts(w3, gov.functions.exec(el.address, call), env["acc"][0])


def test_open_after_attempts_lets_any_live_agent_register(env):
    w3, el, gov = env["w3"], env["el"], env["gov"]
    a0, a2, a3, outsider = (env["agents"][0], env["agents"][2],
                            env["agents"][3], env["acc"][7])
    # a2 holds a millionth of all REP: far below any draw at attempt 2.
    _give_rep(env, a0, 10 ** 6)
    _give_rep(env, a2, 1)
    call = el.encode_abi("setParams", args=[3, 0, 8, 2])
    _send(w3, gov.functions.exec(el.address, call), env["acc"][0])
    _open(env, 1, attempt=2)
    # Open to every live agent that holds REP, however little.
    assert _send(w3, el.functions.registerOption(1, 2, _option(env, "o")),
                 a2).status == 1
    # A live agent with no REP still cannot register.
    assert _reverts(w3, el.functions.registerOption(1, 2, _option(env, "q")), a3)
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


def test_the_draw_doubles_with_each_attempt_and_stays_proportional(env):
    """Small holders are reached gradually: the expected number of proposers
    doubles each attempt, each agent's chance stays its share of REP times
    that number, and only from openAfterAttempts is everyone drawn."""
    w3, el = env["w3"], env["el"]
    whale, small = env["agents"][0], env["agents"][1:5]
    _give_rep(env, whale, 9600)
    for a in small:
        _give_rep(env, a, 100)          # 1% of all REP each
    # Read at a timepoint after every balance above was written.
    w3.provider.ethereum_tester.time_travel(_ts(w3) + 5)
    w3.provider.ethereum_tester.mine_block()
    tp = _ts(w3) - 1
    assert el.functions.openAfterAttempts().call() == 16

    def rate(attempt, n=400):
        hits = sum(el.functions.isDrawn(e, attempt, a, tp).call()
                   for e in range(1, n // len(small) + 1) for a in small)
        return hits / n

    # A 1% holder: 3% at attempt 0, 6%, 12%, 24%, 48%, then certain.
    r0, r2, r4 = rate(0), rate(2), rate(4)
    assert 0.005 <= r0 <= 0.07, r0
    assert 0.06 <= r2 <= 0.19, r2
    assert 0.38 <= r4 <= 0.58, r4
    assert rate(6, n=40) == 1.0
    # The whale (96%) is always drawn; an agent with no REP never is.
    assert all(el.functions.isDrawn(e, 0, whale, tp).call() for e in range(1, 20))
    assert not any(el.functions.isDrawn(1, k, env["acc"][7], tp).call()
                   for k in (0, 5, 16, 80))
    # From openAfterAttempts, a sliver is enough.
    assert el.functions.isDrawn(1, 16, small[0], tp).call()


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
    """A window counts the tool share collected between the previous cutoff
    and this one, so a later anchor of an earlier cutoff gets the same pool."""
    from tests.substrate_chain_helpers import index_tokens, pay_fees
    w3, sub, acc = env["w3"], env["sub"], env["acc"]
    token = index_tokens(sub)[0]

    before_c1 = pay_fees(w3, sub, 10 ** 8, tokens=[token])[token.address.lower()]
    c1 = _ts(w3)
    after_c1 = pay_fees(w3, sub, 10 ** 8, tokens=[token])[token.address.lower()]
    assert anchor_direct(w3, sub, "w1", b"\x01" * 32, "c", b"\x02" * 32,
                         bytes(32), cutoff=c1).status == 1
    assert sub.functions.getAnchor(0).call()[10] == c1
    assert sub.functions.toolShareBetween(token.address, 0, c1).call() == before_c1
    assert sub.functions.toolShareBetween(token.address, c1, _ts(w3)).call() == after_c1
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


# ---------------------------------------------------------------------------
# open(): only the election proposes its elections
# ---------------------------------------------------------------------------

CANCELED, DEFEATED, SUCCEEDED, QUEUED, EXPIRED, EXECUTED = 2, 3, 4, 5, 6, 7


def _set_state(env, pid, state):
    _send(env["w3"], env["gov"].functions.setProposal(pid, state, _ts(env["w3"])),
          env["acc"][0])


def test_open_proposes_the_next_epochs_election_as_the_election_itself(env):
    w3, el, gov = env["w3"], env["el"], env["gov"]
    a0, outsider = env["agents"][0], env["acc"][7]
    # Agents only, and only an epoch that is next (or follows an open one).
    assert _reverts(w3, el.functions.open(1), outsider)
    assert _reverts(w3, el.functions.open(0), a0)
    assert _reverts(w3, el.functions.open(2), a0)
    assert _reverts(w3, el.functions.open(3), a0)

    assert _send(w3, el.functions.open(1), a0).status == 1
    pid = el.functions.proposalIdOf(1, 0).call()
    assert gov.functions.proposerOf(pid).call() == el.address
    assert el.functions.attemptsOpened(1).call() == 1
    assert gov.functions.state(pid).call() == PENDING

    # One live attempt: no second one while the first is pending or voting.
    assert _reverts(w3, el.functions.open(1), a0)
    _set_state(env, pid, ACTIVE)
    assert _reverts(w3, el.functions.open(1), env["agents"][1])
    assert gov.functions.proposeCount().call() == 1


def test_open_numbers_the_next_attempt_itself_once_the_latest_ended(env):
    w3, el, gov = env["w3"], env["el"], env["gov"]
    a0, a1 = env["agents"][:2]
    _send(w3, el.functions.open(1), a0)
    pid0 = el.functions.proposalIdOf(1, 0).call()
    for n, ended in enumerate((DEFEATED, EXPIRED, CANCELED, EXECUTED), start=1):
        latest = el.functions.proposalIdOf(1, n - 1).call()
        # A succeeded attempt is not over: its winner still has to land.
        _set_state(env, latest, SUCCEEDED)
        assert _reverts(w3, el.functions.open(1), a1)
        _set_state(env, latest, ended)
        assert _send(w3, el.functions.open(1), a1).status == 1
        assert el.functions.attemptsOpened(1).call() == n + 1
        nxt = el.functions.proposalIdOf(1, n).call()
        assert gov.functions.proposerOf(nxt).call() == el.address
        assert nxt != pid0
    assert gov.functions.proposeCount().call() == 5


def test_open_allows_one_epoch_ahead_only_once_the_parent_has_a_winner(env):
    """Closes are pipelined: epoch 2's election may run while epoch 1's
    winner sits in the timelock. Not before epoch 1 has a winner, and never
    further ahead, so attempts of future epochs cannot be used up."""
    w3, el = env["w3"], env["el"]
    a0 = env["agents"][0]
    assert _reverts(w3, el.functions.open(2), a0)
    _send(w3, el.functions.open(1), a0)
    pid1 = el.functions.proposalIdOf(1, 0).call()
    for not_decided in (PENDING, ACTIVE, DEFEATED, CANCELED):
        _set_state(env, pid1, not_decided)
        assert _reverts(w3, el.functions.open(2), a0)
    _set_state(env, pid1, SUCCEEDED)
    # Reject won (index 0): decided, but there is no winner to build on.
    _send(w3, env["gov"].functions.setWinner(pid1, True, 0), env["acc"][0])
    assert _reverts(w3, el.functions.open(2), a0)
    _send(w3, env["gov"].functions.setWinner(pid1, True, 1), env["acc"][0])
    assert _reverts(w3, el.functions.open(3), a0)
    assert _send(w3, el.functions.open(2), a0).status == 1
    assert _reverts(w3, el.functions.open(3), a0)
    assert _reverts(w3, el.functions.open(5), a0)
    # Epoch 2 is not next to anchor, so a queued attempt of it never counts
    # as stuck: it is waiting for epoch 1.
    pid2 = el.functions.proposalIdOf(2, 0).call()
    _set_state(env, pid2, QUEUED)
    _send(w3, env["gov"].functions.setEta(pid2, _ts(w3)), env["acc"][0])
    w3.provider.ethereum_tester.time_travel(_ts(w3) + 7200)
    w3.provider.ethereum_tester.mine_block()
    assert _reverts(w3, el.functions.open(2), a0)


def test_open_replaces_a_queued_winner_that_never_executes(env):
    w3, el, gov = env["w3"], env["el"], env["gov"]
    a0 = env["agents"][0]
    _send(w3, el.functions.open(1), a0)
    pid = el.functions.proposalIdOf(1, 0).call()
    _set_state(env, pid, QUEUED)
    # No eta recorded: not stuck.
    assert _reverts(w3, el.functions.open(1), a0)
    eta = _ts(w3) + 100
    _send(w3, gov.functions.setEta(pid, eta), env["acc"][0])
    grace = el.functions.EXECUTE_GRACE().call()
    w3.provider.ethereum_tester.time_travel(eta + grace - 60)
    w3.provider.ethereum_tester.mine_block()
    assert _reverts(w3, el.functions.open(1), a0)
    w3.provider.ethereum_tester.time_travel(eta + grace + 60)
    w3.provider.ethereum_tester.mine_block()
    assert _send(w3, el.functions.open(1), a0).status == 1
    assert el.functions.attemptsOpened(1).call() == 2


def test_an_anchored_epoch_cannot_be_opened_again(env):
    w3, el, gov = env["w3"], env["el"], env["gov"]
    a0 = env["agents"][0]
    _give_rep(env, a0, 1000)
    _send(w3, el.functions.open(1), a0)
    pid = el.functions.proposalIdOf(1, 0).call()
    # Re-point the snapshot so the draw reads REP set above.
    _send(w3, gov.functions.setProposal(pid, PENDING, _ts(w3) + 10), env["acc"][0])
    assert _send(w3, el.functions.registerOption(1, 0, _option(env, "w")), a0).status == 1
    _send(w3, gov.functions.setWinner(pid, True, 1), env["acc"][0])
    call = el.encode_abi("anchorWinner", args=[1, 0])
    assert _send(w3, gov.functions.exec(el.address, call), env["acc"][0]).status == 1
    assert env["sub"].functions.anchorCount().call() == 1
    assert el.functions.lastAnchoredAt().call() > 0
    _set_state(env, pid, EXECUTED)
    assert _reverts(w3, el.functions.open(1), a0)
    assert _send(w3, el.functions.open(2), a0).status == 1


def test_an_attempts_id_cannot_be_known_before_it_is_opened(env):
    """The description, and so the proposal id, is written by open() with
    the opening block's timestamp and randomness: nobody can propose that
    id first. Options and anchoring exist only for opened attempts."""
    w3, el, gov = env["w3"], env["el"], env["gov"]
    a0 = env["agents"][0]
    _give_rep(env, a0, 1000)
    assert el.functions.descriptionOf(1, 0).call() == ""
    unopened = el.functions.proposalIdOf(1, 0).call()
    # Someone holds a proposal under the unopened attempt's would-be id.
    _send(w3, gov.functions.setProposal(unopened, PENDING, _ts(w3) - 1),
          env["acc"][0])
    assert _reverts(w3, el.functions.registerOption(1, 0, _option(env, "x")), a0)
    call = el.encode_abi("anchorWinner", args=[1, 0])
    _send(w3, gov.functions.setWinner(unopened, True, 1), env["acc"][0])
    assert _reverts(w3, gov.functions.exec(el.address, call), env["acc"][0])

    assert _send(w3, el.functions.open(1), a0).status == 1
    desc = el.functions.descriptionOf(1, 0).call()
    assert desc.startswith("Anchor election: epoch 1, attempt 0, opened ")
    opened = el.functions.proposalIdOf(1, 0).call()
    assert opened != unopened
    assert gov.functions.proposerOf(opened).call() == el.address
    # A second attempt gets its own description.
    _set_state(env, opened, DEFEATED)
    assert _send(w3, el.functions.open(1), a0).status == 1
    assert el.functions.descriptionOf(1, 1).call() != desc
