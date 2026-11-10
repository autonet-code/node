"""Decision 2026-10-27 (S0 gap hunt, amended 2026-11-07): a tool mints
only while bonded on ToolBond, to the bond's author, once its review rounds
conclude.
"""

from __future__ import annotations

from nodes.common.federated_reconcile import compute_tool_mint
from nodes.common.voice_state import tool_review_snapshot_from_logs
from tests.test_staked_tool_close import _agent_totals
from tests.test_tool_mint import DIGEST, review, tool_cells

AUTHOR = "0x" + "a0" * 20
E18 = 10 ** 18
R1, R2 = "0x" + "a1" * 20, "0x" + "b2" * 20
SHARES = {R1: 0.3, R2: 0.2}


def _snap(bonds=None, **extra):
    """A top-rated round of two reviewers; ``bonds`` None = no bond data."""
    snap = {"supply": 1000 * E18, "reviewer_share_bps": 4000,
            "tools": {DIGEST: tool_cells()},
            "rounds": {DIGEST: [{"epoch": 1, "supply": 1000 * E18,
                                 "reviews": [review(R1), review(R2)]}]}}
    if bonds is not None:
        snap["bonds"] = bonds
        snap["min_stake"] = 100
    snap.update(extra)
    return snap


def _bonded(author=AUTHOR, stake=100, unbonding=False):
    return _snap({DIGEST: {"author": author, "stake": stake, "unbonding": unbonding}})


def _mint(snapshot, **kw):
    return compute_tool_mint([], registrations=REGS, rep_shares=SHARES,
                             review_snapshot=snapshot, **kw)


REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}


def test_unbonded_tools_do_not_mint_and_the_bond_author_is_paid():
    def mint(snapshot):
        return _agent_totals(_mint(snapshot))
    assert mint(_snap({})) == {}
    assert mint(_bonded(stake=99)) == {}
    assert mint(_bonded(unbonding=True)) == {}
    # The gossiped registration names AUTHOR; the chain's bond names the
    # real author, who is paid (the reviewers are paid either way).
    real = "0x" + "b0" * 20
    assert real in mint(_bonded(author=real))
    assert AUTHOR not in mint(_bonded(author=real))
    # A snapshot without bond data (older deploys, tests) keeps the old rule.
    assert AUTHOR in mint(_snap())


def test_bonds_fold_from_toolbond_logs():
    d = bytes.fromhex(DIGEST)

    def rec(block, event, **args):
        return {"block": block, "index": 0, "event": event, "args": {"digest": d, **args}}
    params = {"block": 1, "index": 0, "event": "ParamsSet", "args": {
        "minStake": 100, "revealWindow": 60, "degradeBps": 1000, "authorSlashBps": 10000,
        "unbondDelay": 1, "reviewerShareBps": 4000}}
    kw = dict(since_block=0, epochs={}, accuracies={}, axis_ids=[])
    logs = [params, rec(2, "Bonded", author=AUTHOR, amount=100, stake=100)]
    snap = tool_review_snapshot_from_logs(logs, block=10, **kw)
    assert snap["bonds"] == {DIGEST: {"author": AUTHOR, "stake": 100, "unbonding": False}}
    assert snap["min_stake"] == 100
    logs.append(rec(3, "UnbondRequested", availableAt=999))
    assert tool_review_snapshot_from_logs(logs, block=10, **kw)["bonds"][DIGEST]["unbonding"]
    logs.append(rec(4, "Bonded", author=AUTHOR, amount=50, stake=150))
    assert tool_review_snapshot_from_logs(logs, block=10, **kw)["bonds"][DIGEST] == {
        "author": AUTHOR, "stake": 150, "unbonding": False}
    logs.append(rec(5, "MalwareSlashed", by=1, bondSlashed=60, repBurned=0))
    assert tool_review_snapshot_from_logs(logs, block=10, **kw)["bonds"][DIGEST]["stake"] == 90
    logs.append(rec(6, "Unbonded", author=AUTHOR, amount=90))
    assert tool_review_snapshot_from_logs(logs, block=10, **kw)["bonds"] == {}


def test_sanctioned_reviewers_weigh_nothing():
    full = _agent_totals(_mint(_bonded()))
    assert full.get(AUTHOR, 0) > 0
    # R2's household is sanctioned: it leaves the tool's weight, so the
    # author's share shrinks. (Its own payout is withheld by the close, after
    # the pool split: see test_sanctions.)
    part = _agent_totals(_mint(_bonded(), sanctioned_households=[R2]))
    assert 0 < part[AUTHOR] < full[AUTHOR]


def test_a_tool_mints_only_once_it_breaks_through():
    """Its reviewers must hold >= A of REP and be >= B of the REP-holding
    households (sized by the GPU sim, 2026-10-28)."""
    others = {f"0x{i:040x}": 0.01 for i in range(1, 97)}     # 96 other holders
    shares = {R1: 0.01, R2: 0.01, **others}

    def mint(gate):
        return _agent_totals(compute_tool_mint(
            [], registrations=REGS, review_snapshot=_bonded(),
            rep_shares=shares, mint_gate=gate))
    assert mint(None).get(AUTHOR, 0) > 0
    # 2 reviewers with 2% of REP, 2 of 98 holders: passes 1% / 0.5%...
    assert mint((100, 50)).get(AUTHOR, 0) > 0
    # ...but not 5% of REP, nor 5% of the holders.
    assert mint((500, 50)) == {}
    assert mint((100, 500)) == {}

