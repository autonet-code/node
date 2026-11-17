"""An election whose winner never executes gets a next attempt, opened
through the election contract (which numbers the attempts), not left queued."""
from types import SimpleNamespace

from nodes.common import anchor_election as ae
from nodes.common.governed_anchor import QUEUED


def _anchorer(now, eta, opened):
    """A driver over stubs. ``opened`` is the list of epochs the daemon asked
    the election contract to open."""
    a = object.__new__(ae.ElectionAnchorer)
    a.state = lambda e: QUEUED
    a.proposal_id = lambda e: 1
    a._now = lambda: now
    a._daemon_key = "0x" + "11" * 32
    a._proposal_args = lambda e: ([], [], [], "")
    a._description_hash = lambda e: b""
    attempts = {"n": 1}

    def _try(e, what, fn, key=None):
        if what == "open":
            opened.append(e.epoch_number)
            attempts["n"] += 1      # the contract accepted: a new attempt
            return True
        return False                # execute keeps reverting

    a._try = _try
    eta_call = SimpleNamespace(call=lambda: eta)
    a._governor = SimpleNamespace(functions=SimpleNamespace(
        proposalEta=lambda pid: eta_call,
        execute=lambda *args: None))
    a._election = SimpleNamespace(functions=SimpleNamespace(
        attemptsOpened=lambda n: SimpleNamespace(call=lambda: attempts["n"]),
        open=lambda n: None))
    return a


def _election():
    return ae.Election(epoch_id="epoch-2-00-1", epoch_number=2, option=())


def test_queued_winner_that_never_executes_gets_a_next_attempt():
    e, opened = _election(), []
    a = _anchorer(now=1000 + ae.EXECUTE_GRACE_S, eta=1000, opened=opened)
    assert a.step(e, []) == "retry"
    assert opened == [2]
    # The attempt number is the contract's, read back after the open.
    assert e.attempt == 1


def test_queued_winner_inside_the_grace_keeps_waiting():
    e, opened = _election(), []
    a = _anchorer(now=1000 + ae.EXECUTE_GRACE_S - 1, eta=1000, opened=opened)
    assert a.step(e, []) != "retry"
    assert opened == [] and e.attempt == 0
