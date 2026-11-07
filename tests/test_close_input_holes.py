"""Holes a code-only read found in the close inputs (2026-10-28): sanctions
skipped some credit paths, and an empty REP read on a chain-backed close
weighed every household 1.0 with the gate off. (Loadout adoption credit and
receipt signatures are gone with the attestation mint, 2026-11-07.)
"""

from __future__ import annotations

from nodes.common.federated_reconcile import compute_tool_mint
from tests.test_signed_receipts_and_bonds import R1, R2, _snap
from tests.test_staked_tool_close import _world
from tests.test_tool_mint import DIGEST

AUTHOR = "0x" + "a0" * 20
REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}


def _weight(**kw):
    out = compute_tool_mint(_world(), [], registrations=REGS,
                            review_snapshot=_snap(), **kw)
    return (out["per_digest"].get(DIGEST) or {}).get("weight", 0.0)


def test_a_sanctioned_household_adds_no_weight():
    shares = {R1: 0.3, R2: 0.2}
    both = _weight(rep_shares=shares)
    one = _weight(rep_shares=shares, sanctioned_households=[R2])
    assert 0 < one < both


def test_a_sanctioned_household_does_not_count_toward_the_gate():
    shares = {R1: 0.3, R2: 0.2}
    # Both reviewers hold 50%; sanction R2 and only 30% remains.
    assert _weight(rep_shares=shares, mint_gate=(5000, 0)) > 0
    assert _weight(rep_shares=shares, mint_gate=(5000, 0),
                   sanctioned_households=[R2]) == 0


def test_empty_rep_on_a_chain_backed_close_mints_nothing():
    # Local regime (no REP read): every household weighs 1.0.
    assert _weight() > 0
    # Chain-backed with no REP anywhere: no weight, gate on, nothing mints.
    assert _weight(rep_shares={}, mint_gate=(100, 50)) == 0
