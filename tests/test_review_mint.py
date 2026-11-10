"""Tool mint from reviews (Decisions 2026-11-07 and 2026-11-09).

A tool is paid once, when its ToolBond rounds conclude: its reviewers clear
the breakthrough gate. Its share of the pool is the REP behind its reviews
(REP / n per household per epoch); reviewers take the reviewer share
whatever the scores, split by accuracy among households above the REP
minimum; the author side is the rest x V x C x S, of which the author keeps
N and the dependencies split 1 - N; a malware verdict leaves the author side
unpaid, also for composites using the tool, and settles at the conclusion
(Decision 2026-11-10: no hold, the DAO clears or designates by proposal);
anything unpaid stays unminted and the pool carries over.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import Keypair
from nodes.common.federated_reconcile import (
    UNMINTED,
    unpaid_key,
    compute_tool_mint,
    federated_epoch_close,
    tool_rating,
    tool_score,
)
from nodes.common.library.charter import DEFAULT_CHARTER

from tests.test_tool_mint import (
    AUTHOR, DIGEST, _batches, _registration_event, review, tool_cells,
)

E18 = 10 ** 18
DEP = "cc" * 32
DEP2 = "ee" * 32
R1, R2, R3 = ("0x" + c * 20 for c in ("a1", "b2", "c3"))
AXES = list(DEFAULT_CHARTER.ids)
REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}
SHARES = {R1: 0.3, R2: 0.2, R3: 0.1, AUTHOR.lower(): 0.4}
COMPOSITE_REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR, "deps": DEP},
                  DEP: {"trust_class": "pinned", "author": "depsmith"}}


def _rev(who: str, acc: Any = E18) -> Dict[str, Any]:
    return review(who, acc)


def _snap(rounds: Dict[str, List[Dict[str, Any]]], *, tools: Any = None,
          **cells: Any) -> Dict[str, Any]:
    t = tools if tools is not None else {d: tool_cells(**cells) for d in rounds}
    return {"supply": 1000 * E18, "reviewer_share_bps": 4000, "tools": t,
            "rounds": rounds}


def _round(epoch: int, *reviews) -> Dict[str, Any]:
    return {"epoch": epoch, "supply": 1000 * E18, "reviews": list(reviews)}


def _totals(out: Dict[str, Any]) -> Dict[str, float]:
    tot: Dict[str, float] = {}
    for bucket in out["node_agent"].values():
        for a, v in bucket.items():
            tot[a] = tot.get(a, 0.0) + v
    return tot


def _mint(rounds, *, state=None, gate=None, **kw):
    return compute_tool_mint(
        [], registrations=kw.pop("regs", REGS),
        rep_shares=kw.pop("shares", SHARES),
        review_snapshot=kw.pop("snapshot", None) or _snap(rounds, **kw.pop("snap", {})),
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


class TestScore:
    """Author side = 60% x V x C x S (Decision 2026-11-09)."""

    def _author(self, **cells):
        out = _mint({DIGEST: [_round(1, _rev(R1))]}, snap=cells)
        return _totals(out), out

    def test_reviewer_paid_whatever_the_scores(self):
        tot, _ = self._author(correctness=-0.5)
        assert tot[R1] == pytest.approx(0.3 * 0.4)
        assert AUTHOR not in tot
        assert tot[UNMINTED] == pytest.approx(0.3 * 0.6)

    def test_correctness_scales_the_author_side(self):
        tot, _ = self._author(correctness=0.5)
        assert tot[AUTHOR] == pytest.approx(0.3 * 0.6 * 0.5)
        assert tot[UNMINTED] == pytest.approx(0.3 * 0.6 * 0.5)

    def test_simplicity_has_a_floor_of_half(self):
        tot, _ = self._author(simplicity=0.0)
        assert tot[AUTHOR] == pytest.approx(0.3 * 0.6 * 0.5)
        tot, _ = self._author(simplicity=0.5)
        assert tot[AUTHOR] == pytest.approx(0.3 * 0.6 * 0.75)

    def test_charter_value_is_one_factor(self):
        tot, _ = self._author(value=0.5)
        assert tot[AUTHOR] == pytest.approx(0.3 * 0.6 * 0.5)
        tot, _ = self._author(value=-0.2)
        assert AUTHOR not in tot

    def test_value_is_the_mean_over_all_axes_together(self):
        cells = tool_cells()
        cells["axes"][AXES[0]] = {"sum": -100 * E18, "mass": E18}
        rating = tool_rating(cells, AXES, 1000 * E18)
        assert rating["value"] == pytest.approx((len(AXES) - 2) / len(AXES))
        assert tool_score(rating) == pytest.approx(rating["value"])

    def test_the_factors_multiply(self):
        tot, out = self._author(correctness=0.8, simplicity=0.5, value=0.5)
        score = 0.5 * 0.8 * 0.75
        assert out["per_digest"][DIGEST]["score"] == pytest.approx(score)
        assert tot[AUTHOR] == pytest.approx(0.3 * 0.6 * score)

    def test_scores_never_change_the_tools_share(self):
        a = _mint({DIGEST: [_round(1, _rev(R1))]}, snap={"correctness": 0.1})
        b = _mint({DIGEST: [_round(1, _rev(R1))]}, snap={"correctness": 1.0})
        assert a["per_digest"][DIGEST]["weight"] == b["per_digest"][DIGEST]["weight"]

    def test_novelty_without_dependencies_leaves_the_rest_unminted(self):
        tot, _ = self._author(novelty=0.25)
        assert tot[AUTHOR] == pytest.approx(0.3 * 0.6 * 0.25)
        assert tot[UNMINTED] == pytest.approx(0.3 * 0.6 * 0.75)


class TestShares:
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
    def test_author_keeps_novelty_dependencies_split_the_rest(self):
        snap = _snap({DIGEST: [_round(1, _rev(R1))]},
                     tools={DIGEST: tool_cells(novelty=0.6)})
        out = _mint(None, regs=COMPOSITE_REGS, snapshot=snap)
        tot = _totals(out)
        side = 0.3 * 0.6
        assert tot[AUTHOR] == pytest.approx(side * 0.6)
        assert tot["depsmith"] == pytest.approx(side * 0.4)

    def test_unrated_composite_dependency_keeps_the_default(self):
        regs = {DIGEST: {"trust_class": "pinned", "author": AUTHOR, "deps": DEP},
                DEP: {"trust_class": "pinned", "author": "depsmith", "deps": DEP2},
                DEP2: {"trust_class": "pinned", "author": "leafsmith"}}
        snap = _snap({DIGEST: [_round(1, _rev(R1))]},
                     tools={DIGEST: tool_cells(novelty=0.5)})
        tot = _totals(_mint(None, regs=regs, snapshot=snap))
        side = 0.3 * 0.6
        assert tot["depsmith"] == pytest.approx(side * 0.5 * 0.7)
        assert tot["leafsmith"] == pytest.approx(side * 0.5 * 0.3)

    def test_a_badly_rated_dependency_still_gets_its_part(self):
        snap = _snap({DIGEST: [_round(1, _rev(R1))]},
                     tools={DIGEST: tool_cells(novelty=0.5),
                            DEP: tool_cells(correctness=-1.0, value=-1.0)})
        tot = _totals(_mint(None, regs=COMPOSITE_REGS, snapshot=snap))
        assert tot["depsmith"] == pytest.approx(0.3 * 0.6 * 0.5)


class TestMalware:
    def test_verdict_leaves_the_author_side_unpaid(self):
        out = _mint({DIGEST: [_round(1, _rev(R1), _rev(R2))]}, snap={"malware": 0.6})
        tot = _totals(out)
        assert AUTHOR not in tot
        assert tot[R1] == pytest.approx(0.5 * 0.4 * 0.5)
        # What the author side would have paid is fixed now under its own
        # key, so the close can record it in ATN; it is never minted.
        assert UNMINTED not in tot
        assert tot[unpaid_key(DIGEST, AUTHOR)] == pytest.approx(0.5 * 0.6)
        assert out["per_digest"][DIGEST]["malware"] is True
        assert out["per_digest"][DIGEST]["unpaid"] == pytest.approx(0.5 * 0.6)
        assert out["per_digest"][DIGEST]["author_mint"] == 0.0

    def test_half_is_not_more_than_half(self):
        out = _mint({DIGEST: [_round(1, _rev(R1))]}, snap={"malware": 0.5})
        assert out["per_digest"][DIGEST]["malware"] is False
        assert _totals(out)[AUTHOR] == pytest.approx(0.3 * 0.6)

    def test_composites_using_a_malware_tool_are_unpaid(self):
        # DEP concluded earlier with a malware verdict.
        state = {"tools": {DEP: {"concluded": True, "rounds": []}}}
        snap = _snap({DIGEST: [_round(2, _rev(R1))]},
                     tools={DIGEST: tool_cells(), DEP: tool_cells(malware=0.9)})
        out = _mint(None, regs=COMPOSITE_REGS, snapshot=snap, state=state)
        tot = _totals(out)
        assert AUTHOR not in tot and "depsmith" not in tot
        assert tot[R1] == pytest.approx(0.3 * 0.4)
        assert out["per_digest"][DIGEST]["malware"] is True

    def test_a_dependencys_flags_count_once_its_reviewers_clear_the_gate(self):
        # DEP's own flags come from too little REP to conclude: no verdict.
        gate = (2500, 0)
        snap = _snap({DIGEST: [_round(1, _rev(R1))], DEP: [_round(1, _rev(R3))]},
                     tools={DIGEST: tool_cells(), DEP: tool_cells(malware=1.0)})
        out = _mint(None, regs=COMPOSITE_REGS, snapshot=snap, gate=gate)
        assert out["per_digest"][DIGEST]["malware"] is False
        assert _totals(out)[AUTHOR] > 0

    def test_a_dependency_too_slashed_to_be_paid_still_has_its_verdict(self):
        # DEP's bond fell below the minimum, so DEP itself never concludes,
        # but its reviewers cleared the gate: composites using it are unpaid.
        snap = _snap({DIGEST: [_round(1, _rev(R1))], DEP: [_round(1, _rev(R2))]},
                     tools={DIGEST: tool_cells(), DEP: tool_cells(malware=1.0)})
        snap["bonds"] = {DIGEST: {"author": AUTHOR.lower(), "stake": 10},
                         DEP: {"author": "depsmith", "stake": 1}}
        snap["min_stake"] = 5
        out = _mint(None, regs=COMPOSITE_REGS, snapshot=snap)
        assert DEP not in out["per_digest"]
        assert out["per_digest"][DIGEST]["malware"] is True


class TestVerdictFlow:
    """The verdict settles at the conclusion; the DAO's word replaces it at
    any time (Decision 2026-11-10)."""

    def _concluded_malware(self):
        out = _mint({DIGEST: [_round(1, _rev(R1), _rev(R2))]}, snap={"malware": 0.6})
        return out, out["tool_rounds_next"]

    def _later(self, state, dao, **kw):
        snap = _snap({}, tools={DIGEST: tool_cells(malware=0.6)})
        snap["verdicts"] = {DIGEST: {"concluded": True, "dao": dao}}
        return _mint(None, snapshot=snap, state=state, **kw)

    def test_the_unpaid_side_is_recorded_and_the_leaf_carries_the_share(self):
        out, state = self._concluded_malware()
        assert out["concluded"] == {DIGEST: 6000}
        assert state["tools"][DIGEST]["unpaid"] == {
            "amounts": {AUTHOR: pytest.approx(0.5 * 0.6)}, "on": [DIGEST]}
        assert state["pruned"] == [DIGEST]

    def test_a_dao_clearing_pays_nothing_in_a_close_and_unprunes(self):
        _, state = self._concluded_malware()
        cleared = self._later(state, False)
        # Paid once: a cleared tool is paid from the treasury, not a close.
        assert cleared["node_agent"] == {} and cleared["per_digest"] == {}
        assert cleared["concluded"] == {}
        assert cleared["pruned"] == []
        # The record stays for the proposal that pays it.
        assert cleared["tool_rounds_next"]["tools"][DIGEST]["unpaid"]["amounts"] == {
            AUTHOR: pytest.approx(0.5 * 0.6)}

    def test_a_dao_designation_needs_no_reviews(self):
        snap = _snap({DIGEST: [_round(1, _rev(R1))]}, tools={DIGEST: tool_cells()})
        snap["verdicts"] = {DIGEST: {"concluded": False, "dao": True}}
        out = _mint(None, snapshot=snap)
        assert out["per_digest"][DIGEST]["malware"] is True
        assert AUTHOR not in _totals(out)
        assert out["pruned"] == [DIGEST]

    def test_a_dao_clearing_replaces_the_reviewers_verdict_before_conclusion(self):
        snap = _snap({DIGEST: [_round(1, _rev(R1))]}, tools={DIGEST: tool_cells(malware=0.9)})
        snap["verdicts"] = {DIGEST: {"concluded": False, "dao": False}}
        out = _mint(None, snapshot=snap)
        assert out["per_digest"][DIGEST]["malware"] is False
        assert _totals(out)[AUTHOR] > 0

    def test_the_leaf_share_is_floored_bps(self):
        out = _mint({DIGEST: [_round(1, _rev(R1))]},
                    snapshot=_snap({DIGEST: [_round(1, _rev(R1))]},
                                   tools={DIGEST: tool_cells(malware=0.0)}))
        assert out["concluded"] == {DIGEST: 0}
        assert out["per_digest"][DIGEST]["malware"] is False


class TestRefusedAndPruned:
    """A registration importing a malware tool is refused; malware tools and
    everything built on them are pruned (Decision 2026-11-10)."""

    MAL_STATE = {"tools": {DEP: {"concluded": True, "rounds": []}}}
    MAL_TOOLS = {DEP: tool_cells(malware=0.9)}

    def _snap(self):
        return _snap({}, tools=dict(self.MAL_TOOLS))

    def test_a_composite_importing_a_malware_tool_is_refused(self):
        ev = _registration_event(1, digest=DIGEST, deps=DEP)
        out = compute_tool_mint(
            [ev], registrations={DEP: COMPOSITE_REGS[DEP]}, rep_shares=SHARES,
            review_snapshot=self._snap(), tool_rounds=self.MAL_STATE)
        assert DIGEST not in out["registrations_next"]
        assert out["refused"] == {DIGEST: [DEP]}
        assert out["pruned"] == sorted([DEP, DIGEST])
        # The refusal is carried, so a later gossip copy stays out.
        again = compute_tool_mint(
            [ev], registrations=out["registrations_next"], rep_shares=SHARES,
            review_snapshot=self._snap(), tool_rounds=out["tool_rounds_next"])
        assert DIGEST not in again["registrations_next"]
        assert again["refused"] == {} and DIGEST in again["pruned"]

    def test_a_transitive_import_is_refused_too(self):
        mid = "dd" * 32
        regs = {DEP: COMPOSITE_REGS[DEP],
                mid: {"trust_class": "pinned", "author": "mid", "deps": DEP}}
        ev = _registration_event(1, digest=DIGEST, deps=mid)
        out = compute_tool_mint(
            [ev], registrations=regs, rep_shares=SHARES,
            review_snapshot=self._snap(), tool_rounds=self.MAL_STATE)
        assert out["refused"] == {DIGEST: [DEP]}
        assert out["pruned"] == sorted([DEP, mid, DIGEST])

    def test_a_clean_import_registers(self):
        ev = _registration_event(1, digest=DIGEST, deps=DEP)
        out = compute_tool_mint(
            [ev], registrations={DEP: COMPOSITE_REGS[DEP]}, rep_shares=SHARES,
            review_snapshot=_snap({}, tools={DEP: tool_cells()}))
        assert DIGEST in out["registrations_next"]
        assert out["refused"] == {} and out["pruned"] == []

    def test_a_composite_registered_before_the_verdict_is_pruned_after(self):
        out = _mint(None, regs=COMPOSITE_REGS, snapshot=self._snap(),
                    state=self.MAL_STATE)
        assert out["pruned"] == sorted([DEP, DIGEST])

    def test_a_dao_clearing_unprunes_the_composites(self):
        snap = self._snap()
        snap["verdicts"] = {DEP: {"concluded": True, "dao": False}}
        out = _mint(None, regs=COMPOSITE_REGS, snapshot=snap, state=self.MAL_STATE)
        assert out["pruned"] == []


class TestRatingsCarry:
    def test_ratings_are_plain_means(self):
        cells = tool_cells(correctness=0.9, simplicity=0.4, novelty=0.3,
                           malware=0.25, value=0.5, mass=300 * E18)
        out = _mint({DIGEST: [_round(1, _rev(R1))]}, snapshot=_snap(
            {DIGEST: [_round(1, _rev(R1))]}, tools={DIGEST: cells}))
        r = out["ratings_next"][DIGEST]
        assert r["correctness"] == pytest.approx(0.9)
        assert r["simplicity"] == pytest.approx(0.4)
        assert r["novelty"] == pytest.approx(0.3)
        assert r["malware"] == pytest.approx(0.25)
        assert r["value"] == pytest.approx(0.5)
        assert r["axes"] == {a: pytest.approx(0.5) for a in AXES}
        assert r["mass"] == pytest.approx(0.3)

    def test_unrated_tool_has_no_means(self):
        out = compute_tool_mint([], registrations=REGS)
        assert out["ratings_next"][DIGEST] == {
            "correctness": None, "simplicity": None, "novelty": None,
            "malware": None, "axes": {}, "value": None, "mass": 0.0}


class TestCloseCarry:
    def test_unminted_remainder_carries_into_next_pool(self):
        kp = Keypair.generate()
        canonical = canonical_order(_batches([_registration_event(1)], kp))
        snap = _snap({DIGEST: [_round(1, _rev(R1))]}, correctness=0.0)
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

    def test_tool_mint_is_in_pool_units(self):
        """The close's per-tool mint, reviewer mint and author mint are all
        in ATN, the units of agent_mint (the app shows them side by side)."""
        kp = Keypair.generate()
        canonical = canonical_order(_batches([_registration_event(1)], kp))
        snap = _snap({DIGEST: [_round(1, _rev(R1))]})
        out = federated_epoch_close(
            canonical, emission_pool=10.0,
            tool_registrations=REGS, rep_shares=SHARES, review_snapshot=snap)
        e = out["tool_mint"][DIGEST]
        assert e["reviewer_mint"] == pytest.approx(4.0)
        assert e["author_mint"] + e["reviewer_mint"] == pytest.approx(e["mint"])
        assert e["mint"] == pytest.approx(sum(out["agent_mint"].values()), abs=1e-5)

    def test_deterministic(self):
        a = _mint({DIGEST: [_round(1, _rev(R1), _rev(R2))]})
        b = _mint({DIGEST: [_round(1, _rev(R2), _rev(R1))]})
        assert a["node_agent"] == b["node_agent"]


class TestUnpaidAuthorSide:
    """The author side a malware verdict leaves unpaid is fixed in ATN at the
    conclusion, recorded, and left unminted: it carries over, and nothing is
    reserved (Decision 2026-11-10)."""

    def _conclude(self, pool=10.0):
        kp = Keypair.generate()
        canonical = canonical_order(_batches([_registration_event(1)], kp))
        snap = _snap({DIGEST: [_round(1, _rev(R1), _rev(R2))]}, malware=0.6)
        return canonical, federated_epoch_close(
            canonical, emission_pool=pool, tool_registrations=REGS,
            rep_shares=SHARES, review_snapshot=snap)

    def _next(self, canonical, first, dao, pool=3.0):
        snap = _snap({}, tools={DIGEST: tool_cells(malware=0.6)})
        snap["verdicts"] = {DIGEST: {"concluded": True, "dao": dao}}
        return federated_epoch_close(
            canonical, emission_pool=pool, tool_registrations=REGS,
            rep_shares=SHARES, review_snapshot=snap,
            tool_rounds=first["tool_rounds"])

    def test_fixed_in_atn_recorded_and_carried(self):
        _, first = self._conclude()
        # One tool takes the whole 10 ATN pool: 4 to its reviewers; the
        # author side, 6 x score 1, is recorded and carried over.
        assert first["malware_unpaid"] == {DIGEST: {"amounts": {AUTHOR: 6.0}, "on": [DIGEST]}}
        assert first["tool_rounds"]["tools"][DIGEST]["unpaid"]["amounts"] == {AUTHOR: 6.0}
        assert sum(first["agent_mint"].values()) == pytest.approx(4.0)
        assert first["tool_rounds"]["pool_carry"] == 6_000_000
        assert first["tool_mint"][DIGEST]["unpaid"] == pytest.approx(6.0)
        assert first["pruned"] == [DIGEST]

    def test_a_later_close_spends_the_carry(self):
        canonical, first = self._conclude()
        later = self._next(canonical, first, None)
        assert later["agent_mint"] == {} and later["malware_unpaid"] == {}
        assert later["tool_rounds"]["pool_carry"] == 9_000_000

    def test_a_clearing_mints_nothing_and_keeps_the_record(self):
        canonical, first = self._conclude()
        cleared = self._next(canonical, first, False)
        assert cleared["agent_mint"] == {}
        assert cleared["tool_rounds"]["pool_carry"] == 9_000_000
        assert cleared["tool_rounds"]["tools"][DIGEST]["unpaid"]["amounts"] == {AUTHOR: 6.0}
        assert cleared["pruned"] == []

    def test_the_carry_bundle_holds_the_record_and_the_pruned_list(self):
        from nodes.common.authoritative_encoding import (
            decode_carry_bundle, encode_carry_bundle)
        _, first = self._conclude()
        blob = encode_carry_bundle({"tool_registrations": first["tool_registrations"],
                                    "tool_rounds": first["tool_rounds"],
                                    "tool_ratings": first["tool_ratings"],
                                    "service_reviews": first["service_reviews"],
                                    "service_ratings": first["service_ratings"]},
                                   epoch_id="e1")
        carry = decode_carry_bundle(blob)["carry"]
        assert carry["tool_rounds"]["tools"][DIGEST]["unpaid"]["amounts"] == {AUTHOR: 6.0}
        assert carry["tool_rounds"]["pruned"] == [DIGEST]

    def test_conservation_across_closes(self):
        canonical, first = self._conclude()
        for dao in (None, False, True):
            nxt = self._next(canonical, first, dao)
            minted = sum(round(v * 1_000_000) for d in (first, nxt)
                         for v in d["agent_mint"].values())
            assert minted + nxt["tool_rounds"]["pool_carry"] == 13_000_000


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
