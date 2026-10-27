"""Decision 2026-10-27 (S0 gap hunt): on a chain-backed close an attested
receipt counts only when the agent it names signed it, and a tool mints only
while bonded on ToolBond, to the bond's author.
"""

from __future__ import annotations

from eth_account import Account

from nodes.common.federated_reconcile import compute_tool_mint
from nodes.common.tool_receipts import sign_receipt, verified_receipts
from nodes.common.voice_state import tool_review_snapshot_from_logs
from tests.test_staked_tool_close import _agent_totals, _world
from tests.test_tool_mint import DIGEST, _receipt_event

DOMAIN = "31337:0x" + "da" * 20
ALICE = Account.from_key("0x" + "11" * 32)
BOB = Account.from_key("0x" + "22" * 32)
AUTHOR = "0x" + "a0" * 20


def _signed(seq, acct, *, domain=DOMAIN):
    ev = _receipt_event(seq, acct.address)
    ev["agent_sig"] = sign_receipt(acct.key.hex(), domain, ev)
    return ev


def _bonded(author=AUTHOR, stake=100, unbonding=False):
    return {"bonds": {DIGEST: {"author": author, "stake": stake, "unbonding": unbonding}},
            "min_stake": 100}


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


def test_forged_receipts_mint_nothing():
    honest = compute_tool_mint(_world(), [_signed(1, ALICE), _signed(2, BOB)],
                               registrations=REGS, review_snapshot=_bonded(),
                               receipt_domain=DOMAIN)
    forged = []
    for i, acct in enumerate((ALICE, BOB)):
        ev = _receipt_event(i + 1, acct.address)
        ev["agent_sig"] = sign_receipt("0x" + "33" * 32, DOMAIN, ev)
        forged.append(ev)
    none = compute_tool_mint(_world(), forged, registrations=REGS,
                             review_snapshot=_bonded(), receipt_domain=DOMAIN)
    assert _agent_totals(honest).get(AUTHOR, 0) > 0
    assert _agent_totals(none) == {}


def test_unbonded_tools_do_not_mint_and_the_bond_author_is_paid():
    receipts = [_signed(1, ALICE), _signed(2, BOB)]

    def mint(snapshot):
        return _agent_totals(compute_tool_mint(
            _world(), receipts, registrations=REGS, review_snapshot=snapshot,
            receipt_domain=DOMAIN))
    assert mint({"bonds": {}, "min_stake": 100}) == {}
    assert mint(_bonded(stake=99)) == {}
    assert mint(_bonded(unbonding=True)) == {}
    # The gossiped registration names AUTHOR; the chain's bond names the
    # real author, who is paid.
    real = "0x" + "b0" * 20
    assert set(mint(_bonded(author=real))) == {real}
    # A snapshot without bond data (older deploys, tests) keeps the old rule.
    assert set(mint({})) == {AUTHOR}


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


def test_denied_tools_and_sanctioned_usage_mint_nothing():
    receipts = [_signed(1, ALICE), _signed(2, BOB)]

    def mint(**kw):
        return _agent_totals(compute_tool_mint(
            _world(), receipts, registrations=REGS, review_snapshot=_bonded(),
            receipt_domain=DOMAIN, **kw))
    full = mint()
    assert full.get(AUTHOR, 0) > 0
    assert mint(denied_tools=["0x" + DIGEST]) == {}
    # Bob's household is sanctioned: only Alice's use credits the author.
    part = mint(sanctioned_households=[BOB.address.lower()])
    assert 0 < part[AUTHOR] < full[AUTHOR]


def test_the_close_denies_what_containment_denies():
    from nodes.common.containment import CONTAINMENT_DEFAULTS, denied_digests
    snap = {"tools": {"aa" * 32: {"harm": {"sum": 80 * 300, "mass": 300}}},
            "cooling_harm": {"bb" * 32: {"sum": 90 * 300, "mass": 300},
                             "cc" * 32: {"sum": 90 * 50, "mass": 50}}}
    # cc: harm from too little REP (50 of 10,000 = 50 bps < 200).
    assert denied_digests(snap, 10_000, dict(CONTAINMENT_DEFAULTS)) == ["aa" * 32, "bb" * 32]
