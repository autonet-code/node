"""Tool mint from staked reviews (Decision 2026-11-07).

A tool is paid once, when its ToolBond rounds conclude: its reviewers clear
the breakthrough gate. Its share of the pool is the REP behind its reviews
(REP / n per household per epoch); reviewers take the reviewer share
whatever the verdict, split by accuracy among households above the REP
minimum; the author takes the rest scaled by the rating, fanned over the
dependencies; anything unpaid stays unminted and the pool carries over.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import Keypair
from nodes.common.federated_reconcile import (
    UNMINTED,
    compute_tool_mint,
    federated_epoch_close,
)
from nodes.common.library.charter import DEFAULT_CHARTER

from tests.test_tool_mint import AUTHOR, DIGEST, EMBED_DIM, _batches, _registration_event

E18 = 10 ** 18
DEP = "cc" * 32
R1, R2, R3 = ("0x" + c * 20 for c in ("a1", "b2", "c3"))
AXES = list(DEFAULT_CHARTER.ids)
REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}
SHARES = {R1: 0.3, R2: 0.2, R3: 0.1, AUTHOR.lower(): 0.4}


def _rev(who: str, acc: Any = E18) -> Dict[str, Any]:
    return {"reviewer": who, "stake": 0, "accuracy": acc,
            "axes": ["correctness"], "harm": False}


def _snap(rounds: Dict[str, List[Dict[str, Any]]], *, rating: float = 1.0,
          tools: Any = None) -> Dict[str, Any]:
    t = tools if tools is not None else {
        d: {"axes": {a: {"sum": int(rating * 100 * E18), "mass": E18} for a in AXES},
            "harm": {"sum": 0, "mass": 0}} for d in rounds}
    return {"supply": 1000 * E18, "reviewer_share_bps": 4000, "tools": t,
            "rounds": rounds}


def _round(epoch: int, *reviews) -> Dict[str, Any]:
    return {"epoch": epoch, "supply": 1000 * E18, "prior_mass": {},
            "reviews": list(reviews)}


def _totals(out: Dict[str, Any]) -> Dict[str, float]:
    tot: Dict[str, float] = {}
    for bucket in out["node_agent"].values():
        for a, v in bucket.items():
            tot[a] = tot.get(a, 0.0) + v
    return tot


def _mint(rounds, *, state=None, gate=None, **kw):
    return compute_tool_mint(
        [], registrations=kw.pop("regs", REGS),
        rep_shares=kw.pop("shares", SHARES), review_snapshot=_snap(rounds, **kw.pop("snap", {})),
        tool_rounds=state, mint_gate=gate, **kw)


class TestConclusion:
    def test_concludes_once_and_pays_reviewers_and_author(self):
        out = _mint({DIGEST: [_round(1, _rev(R1), _rev(R2))]})
        tot = _totals(out)
        w = 0.3 + 0.2
        assert out["per_digest"][DIGEST]["weight"] == pytest.approx(w)
        assert tot[R1] == pytest.approx(w * 0.4 * 0.5)
        assert tot[R2] == pytest.approx(w * 0.4 * 0.5)
        assert tot[AUTHOR] == pytest.approx(w * 0.6)
        assert UNMINTED not in tot
        assert out["tool_rounds_next"]["tools"][DIGEST]["concluded"] is True
        # A later round pays nothing more.
        again = _mint({DIGEST: [_round(2, _rev(R3))]}, state=out["tool_rounds_next"])
        assert again["node_agent"] == {}

    def test_attestations_count_for_nothing(self):
        out = compute_tool_mint([], registrations=REGS, rep_shares=SHARES)
        assert out["per_digest"] == {}

    def test_below_gate_carries_then_concludes(self):
        gate = (6000, 0)   # reviewers must hold 60% of REP
        first = _mint({DIGEST: [_round(1, _rev(R1))]}, gate=gate)
        assert first["per_digest"] == {}
        assert first["tool_rounds_next"]["tools"][DIGEST]["concluded"] is False
        second = _mint({DIGEST: [_round(2, _rev(R2), _rev(R3))]}, gate=gate,
                       state=first["tool_rounds_next"])
        assert second["per_digest"][DIGEST]["weight"] == pytest.approx(0.6)
        tot = _totals(second)
        assert set(tot) >= {R1, R2, R3, AUTHOR}

    def test_decay_fades_carried_rounds(self):
        gate = (6000, 0)
        first = _mint({DIGEST: [_round(1, _rev(R1))]}, gate=gate, review_decay=0.5)
        # R1's 0.3 fades to 0.15: 0.15 + 0.2 + 0.1 < 0.6, still open.
        second = _mint({DIGEST: [_round(2, _rev(R2), _rev(R3))]}, gate=gate,
                       state=first["tool_rounds_next"], review_decay=0.5)
        assert second["per_digest"] == {}

    def test_household_gate_counts_distinct_households(self):
        out = _mint({DIGEST: [_round(1, _rev(R1))]}, gate=(0, 5000))
        assert out["per_digest"] == {}   # 1 of 4 holders < 50%


class TestShares:
    def test_reviewer_paid_whatever_the_verdict(self):
        out = _mint({DIGEST: [_round(1, _rev(R1))]}, snap={"rating": -0.5})
        tot = _totals(out)
        assert tot[R1] == pytest.approx(0.3 * 0.4)
        assert AUTHOR not in tot
        assert tot[UNMINTED] == pytest.approx(0.3 * 0.6)

    def test_author_scaled_by_rating(self):
        out = _mint({DIGEST: [_round(1, _rev(R1))]}, snap={"rating": 0.5})
        tot = _totals(out)
        assert tot[AUTHOR] == pytest.approx(0.3 * 0.6 * 0.5)
        assert tot[UNMINTED] == pytest.approx(0.3 * 0.6 * 0.5)

    def test_split_by_accuracy_alone(self):
        out = _mint({DIGEST: [_round(1, _rev(R1, E18), _rev(R2, E18 // 3))]})
        tot = _totals(out)
        assert tot[R1] / tot[R2] == pytest.approx(3.0)

    def test_min_rep_excludes_small_reviewers(self):
        out = _mint({DIGEST: [_round(1, _rev(R1), _rev(R3))]},
                    reviewer_min_rep_bps=2000)
        tot = _totals(out)
        assert R3 not in tot
        assert tot[R1] == pytest.approx(0.4 * 0.4)

    def test_many_reviews_split_a_households_rep(self):
        # R1 reviews two tools in epoch 1: each counts 0.3 / 2.
        regs = {**REGS, DEP: {"trust_class": "pinned", "author": "other"}}
        out = _mint({DIGEST: [_round(1, _rev(R1))], DEP: [_round(1, _rev(R1))]}, regs=regs)
        assert out["per_digest"][DIGEST]["weight"] == pytest.approx(0.15)
        assert out["per_digest"][DEP]["weight"] == pytest.approx(0.15)

    def test_author_household_excluded(self):
        out = _mint({DIGEST: [_round(1, _rev(AUTHOR.lower()), _rev(R1))]})
        assert out["per_digest"][DIGEST]["weight"] == pytest.approx(0.3)
        assert AUTHOR.lower() not in {a.lower() for a in _totals(out) if a != AUTHOR}


class TestComposition:
    def test_author_side_fans_to_dependency(self):
        regs = {DIGEST: {"trust_class": "pinned", "author": AUTHOR, "deps": DEP},
                DEP: {"trust_class": "pinned", "author": "depsmith"}}
        out = _mint({DIGEST: [_round(1, _rev(R1))]}, regs=regs)
        tot = _totals(out)
        side = 0.3 * 0.6
        assert tot[AUTHOR] == pytest.approx(side * 0.7)
        assert tot["depsmith"] == pytest.approx(side * 0.3)

    def test_denial_passes_up_to_composites(self):
        regs = {DIGEST: {"trust_class": "pinned", "author": AUTHOR, "deps": DEP},
                DEP: {"trust_class": "pinned", "author": "depsmith"}}
        out = _mint({DIGEST: [_round(1, _rev(R1))]}, regs=regs, denied_tools=[DEP])
        assert out["per_digest"] == {}


class TestCloseCarry:
    def test_unminted_remainder_carries_into_next_pool(self):
        kp = Keypair.generate()
        canonical = canonical_order(_batches([_registration_event(1)], kp))
        snap = _snap({DIGEST: [_round(1, _rev(R1))]}, rating=0.0)
        out = federated_epoch_close(
            canonical, emission_pool=10.0,
            tool_registrations=REGS, rep_shares=SHARES, review_snapshot=snap)
        # Only the reviewer share mints: 40% of the pool; 60% carries.
        assert sum(out["agent_mint"].values()) == pytest.approx(4.0)
        assert out["tool_rounds"]["pool_carry"] == 6_000_000
        nothing = federated_epoch_close(
            canonical, emission_pool=1.0,
            tool_registrations=REGS, rep_shares=SHARES,
            review_snapshot=_snap({}), tool_rounds=out["tool_rounds"])
        assert nothing["agent_mint"] == {}
        assert nothing["tool_rounds"]["pool_carry"] == 7_000_000

    def test_deterministic(self):
        a = _mint({DIGEST: [_round(1, _rev(R1), _rev(R2))]})
        b = _mint({DIGEST: [_round(1, _rev(R2), _rev(R1))]})
        assert a["node_agent"] == b["node_agent"]


class TestSanctionedReviewer:
    def test_cut_stays_unminted_not_redistributed(self):
        out = _mint({DIGEST: [_round(1, _rev(R1), _rev(R2))]},
                    sanctioned_households=[R2])
        tot = _totals(out)
        assert R2 not in tot
        # R2 adds no weight, and its accuracy share of the reviewer pool
        # stays unminted instead of going to R1.
        assert out["per_digest"][DIGEST]["weight"] == pytest.approx(0.3)
        assert tot[R1] == pytest.approx(0.3 * 0.4 * 0.5)
