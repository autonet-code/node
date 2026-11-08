"""Decision 2026-10-27 (S0 gap hunt, amended 2026-11-07): signed receipts
still verify (they no longer mint), and a tool mints only while bonded on
ToolBond, to the bond's author, once its review rounds conclude.
"""

from __future__ import annotations

from eth_account import Account

from nodes.common.federated_reconcile import compute_tool_mint
from nodes.common.tool_receipts import sign_receipt, verified_receipts
from nodes.common.voice_state import tool_review_snapshot_from_logs
from nodes.common.library.charter import DEFAULT_CHARTER
from tests.test_staked_tool_close import _agent_totals
from tests.test_tool_mint import DIGEST, _receipt_event

DOMAIN = "31337:0x" + "da" * 20
ALICE = Account.from_key("0x" + "11" * 32)
BOB = Account.from_key("0x" + "22" * 32)
AUTHOR = "0x" + "a0" * 20


def _signed(seq, acct, *, domain=DOMAIN):
    ev = _receipt_event(seq, acct.address)
    ev["agent_sig"] = sign_receipt(acct.key.hex(), domain, ev)
    return ev


E18 = 10 ** 18
R1, R2 = "0x" + "a1" * 20, "0x" + "b2" * 20
AXES = list(DEFAULT_CHARTER.ids)
SHARES = {R1: 0.3, R2: 0.2}


def _snap(bonds=None, **extra):
    """A rating-1.0 round of two reviewers; ``bonds`` None = no bond data."""
    revs = [{"reviewer": r, "stake": 0, "accuracy": E18, "axes": ["correctness"],
             "harm": False} for r in (R1, R2)]
    snap = {"supply": 1000 * E18, "reviewer_share_bps": 4000,
            "tools": {DIGEST: {"axes": {a: {"sum": 100 * E18, "mass": E18} for a in AXES},
                               "harm": {"sum": 0, "mass": 0}}},
            "rounds": {DIGEST: [{"epoch": 1, "supply": 1000 * E18,
                                 "prior_mass": {}, "reviews": revs}]}}
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


def test_only_receipts_signed_by_their_agent_count():
    forged = _receipt_event(3, BOB.address)
    forged["agent_sig"] = sign_receipt(ALICE.key.hex(), DOMAIN, forged)
    other_domain = _signed(4, ALICE, domain="1:0x" + "ee" * 20)
    unsigned = _receipt_event(5, ALICE.address)
    events = [_signed(1, ALICE), _signed(2, BOB), forged, other_domain, unsigned]
    kept = verified_receipts(events, DOMAIN)
    assert [e["seq"] for e in kept] == [1, 2]
    # No domain (no chain, tests): nothing is checked.
    assert len(verified_receipts(events, None)) == 5


def test_a_receipt_counts_once():
    ev = _signed(1, ALICE)
    assert len(verified_receipts([ev, dict(ev), dict(ev)], DOMAIN)) == 1


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
        "unbondDelay": 1, "reviewerShareBps": 4000, "harmWeightBps": 20000}}
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
    logs.append(rec(5, "Finalized", epoch=1, harm=0, bondSlashed=60, lossPool=0, repPool=0))
    assert tool_review_snapshot_from_logs(logs, block=10, **kw)["bonds"][DIGEST]["stake"] == 90
    logs.append(rec(6, "Unbonded", author=AUTHOR, amount=90))
    assert tool_review_snapshot_from_logs(logs, block=10, **kw)["bonds"] == {}


def test_denied_tools_and_sanctioned_reviewers_weigh_nothing():
    full = _agent_totals(_mint(_bonded()))
    assert full.get(AUTHOR, 0) > 0
    assert _agent_totals(_mint(_bonded(), denied_tools=["0x" + DIGEST])) == {}
    # R2's household is sanctioned: it leaves the tool's weight, so the
    # author's share shrinks. (Its own payout is withheld by the close, after
    # the pool split: see test_sanctions.)
    part = _agent_totals(_mint(_bonded(), sanctioned_households=[R2]))
    assert 0 < part[AUTHOR] < full[AUTHOR]


def test_the_close_denies_what_containment_denies():
    from nodes.common.containment import CONTAINMENT_DEFAULTS, denied_digests
    snap = {"tools": {"aa" * 32: {"harm": {"sum": 80 * 300, "mass": 300}}},
            "cooling_harm": {"bb" * 32: {"sum": 90 * 300, "mass": 300},
                             "cc" * 32: {"sum": 90 * 5, "mass": 5}}}
    # cc: harm from too little REP (5 of 10,000 = 5 bps < 10).
    assert denied_digests(snap, 10_000, dict(CONTAINMENT_DEFAULTS)) == ["aa" * 32, "bb" * 32]


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
