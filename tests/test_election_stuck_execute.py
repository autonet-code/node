"""An election whose winner never executes is re-posted, not left queued."""
from types import SimpleNamespace

from nodes.common import anchor_election as ae
from nodes.common.governed_anchor import QUEUED


def _anchorer(now, eta):
    a = object.__new__(ae.ElectionAnchorer)
    a.state = lambda e: QUEUED
    a.proposal_id = lambda e: 1
    a._now = lambda: now
    a._try = lambda *args, **kw: False          # execute keeps reverting
    a._proposal_args = lambda e: ([], [], [], "")
    a._description_hash = lambda e: b""
    eta_call = SimpleNamespace(call=lambda: eta)
    a._governor = SimpleNamespace(functions=SimpleNamespace(
        proposalEta=lambda pid: eta_call,
        execute=lambda *args: None))
    return a


def _election():
    return ae.Election(epoch_id="epoch-2-00-1", epoch_number=2, option=())


def test_queued_winner_that_never_executes_is_reposted():
    e = _election()
    assert _anchorer(now=1000 + ae.EXECUTE_GRACE_S, eta=1000).step(e, []) == "retry"
    assert e.attempt == 1


def test_queued_winner_inside_the_grace_keeps_waiting():
    e = _election()
    assert _anchorer(now=1000 + ae.EXECUTE_GRACE_S - 1, eta=1000).step(e, []) != "retry"
    assert e.attempt == 0
