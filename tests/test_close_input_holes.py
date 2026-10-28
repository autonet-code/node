"""Holes a code-only read found in the close inputs (2026-10-28): loadout
adoption credit skipped receipt signatures and sanctions, and an empty REP
read on a chain-backed close weighed every household 1.0 with the gate off.
"""

from __future__ import annotations

from eth_account import Account

from nodes.common.federated_reconcile import compute_tool_mint
from nodes.common.tool_receipts import sign_receipt
from tests.test_staked_tool_close import _world
from tests.test_tool_mint import DIGEST, _receipt_event

DOMAIN = "31337:0x" + "da" * 20
DISTRO = "22" * 32
ALICE = Account.from_key("0x" + "11" * 32)
BOB = Account.from_key("0x" + "22" * 32)
REGS = {DIGEST: {"trust_class": "pinned", "author": "0x" + "a0" * 20},
        DISTRO: {"trust_class": "pinned", "author": "0x" + "d0" * 20}}


def _loadout_receipt(seq, acct, *, sign=True):
    ev = _receipt_event(seq, acct.address, loadout=DISTRO)
    if sign:
        ev["agent_sig"] = sign_receipt(acct.key.hex(), DOMAIN, ev)
    return ev


def _distro_mint(events, **kw):
    out = compute_tool_mint(_world(), events, registrations=REGS,
                            receipt_domain=DOMAIN, **kw)
    return (out["per_digest"].get(DISTRO) or {}).get("mint", 0.0)


def test_unsigned_loadout_receipts_credit_no_adoption():
    assert _distro_mint([_loadout_receipt(1, ALICE), _loadout_receipt(2, BOB)]) > 0
    assert _distro_mint([_loadout_receipt(1, ALICE, sign=False),
                         _loadout_receipt(2, BOB, sign=False)]) == 0


def test_a_sanctioned_household_adopts_for_nobody():
    events = [_loadout_receipt(1, ALICE), _loadout_receipt(2, BOB)]
    both = _distro_mint(events)
    one = _distro_mint(events, sanctioned_households=[BOB.address])
    assert 0 < one < both


def test_empty_rep_on_a_chain_backed_close_mints_nothing():
    receipts = []
    for i, acct in enumerate((ALICE, BOB)):
        ev = _receipt_event(i + 1, acct.address)
        ev["agent_sig"] = sign_receipt(acct.key.hex(), DOMAIN, ev)
        receipts.append(ev)

    def total(**kw):
        out = compute_tool_mint(_world(), receipts, registrations=REGS,
                                receipt_domain=DOMAIN, **kw)
        return sum(v.get("mint", 0.0) for v in out["per_digest"].values())
    # Chain-free regime (None): every household weighs 1.0.
    assert total() > 0
    # Chain-backed with no REP anywhere: no weight, gate on, nothing mints.
    assert total(voice_weights={}, rep_shares={}, mint_gate=(100, 50)) == 0
