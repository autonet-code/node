"""DAO sanctions (docs/accountability.md): the pinned Registry read, the
household a sanction names, reviews that move no rating, and the close
withholding a sanctioned household's mint without passing it to others.
"""

from __future__ import annotations

import json

import pytest

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import Keypair
from nodes.common.federated_reconcile import federated_epoch_close
from nodes.common.voice_state import (
    parse_sanction,
    sanctioned_households,
    sanctions_from_records,
)

from tests.test_staked_tool_close import (
    AUD_A,
    AUD_B,
    DIGEST,
    _batches,
    _fold,
    _registration_event,
)
from tests import test_staked_tool_close as stc
from tests.test_signed_receipts_and_bonds import _snap
from tests.test_tool_mint import AUTHOR

E18 = 10 ** 18
HOUSE = "0x" + "aa" * 20
AGENT = "0x" + "a1" * 20


def _write(block, key, value, index=0):
    return {"block": block, "index": index, "key": key, "value": value}


class TestRegistryRead:
    def test_last_write_wins_and_empty_lifts(self):
        recs = [
            _write(1, f"sanction.{HOUSE}", json.dumps({"reason": "r1"})),
            _write(2, f"sanction.{AUD_B}", json.dumps({"reason": "r2"})),
            _write(3, f"sanction.{AUD_B}", ""),
            _write(4, "jurisdiction.autonet.charter", "{}"),
        ]
        assert sanctions_from_records(recs, 10, 0) == {HOUSE: {"reason": "r1"}}

    def test_pinned_block_ignores_later_writes(self):
        recs = [_write(1, f"sanction.{HOUSE}", "{}"),
                _write(5, f"sanction.{HOUSE}", "")]
        assert HOUSE in sanctions_from_records(recs, 4, 0)
        assert sanctions_from_records(recs, 5, 0) == {}

    def test_until_ends_it_on_its_own(self):
        recs = [_write(1, f"sanction.{HOUSE}", json.dumps({"until": 1000}))]
        assert HOUSE in sanctions_from_records(recs, 10, 999)
        assert sanctions_from_records(recs, 10, 1000) == {}

    def test_address_case_is_normalized(self):
        recs = [_write(1, "sanction." + HOUSE.upper().replace("0X", "0x"), "{}")]
        assert list(sanctions_from_records(recs, 10, 0)) == [HOUSE]

    def test_values_the_dao_wrote_loosely_still_sanction(self):
        assert parse_sanction("   ") is None
        assert parse_sanction("fraud, see forum") == {"note": "fraud, see forum"}
        assert parse_sanction('{"until": "soon"}') == {}
        assert parse_sanction('{"until": true}') == {}

    def test_a_named_agent_sanctions_its_household(self):
        owner_of = {AGENT: HOUSE}
        assert sanctioned_households([AGENT], owner_of) == [HOUSE]
        assert sanctioned_households([AUD_B], owner_of) == [AUD_B]

    def test_rotating_away_from_a_sanction_carries_it(self):
        """Survey 2026-10-26: rotating a sanctioned household's agent to a
        fresh wallet used to lift the sanction. The new owner signed for
        the agent, so its household is sanctioned from the rotation on,
        and so is the next one it rotates to. A rotation before the
        sanction was written carries nothing."""
        fresh, third = "0x" + "bb" * 20, "0x" + "cc" * 20
        # AGENT moved HOUSE -> fresh at block 20, then fresh -> third at 30.
        owner_of = {AGENT: third}
        rotations = [(20, AGENT, HOUSE, fresh), (30, AGENT, fresh, third)]
        got = sanctioned_households([HOUSE], owner_of, rotations, {HOUSE: 10})
        assert got == sorted({HOUSE, fresh, third})
        # Sold before the sanction was written: nothing carries.
        sold_before = sanctioned_households([HOUSE], owner_of, rotations,
                                            {HOUSE: 40})
        assert sold_before == [HOUSE]


class TestReviews:
    def test_sanctioned_reviews_move_no_rating_but_keep_their_round(self):
        recs = stc.TestReviewSnapshotFold()._round1()
        plain = _fold(recs)
        from nodes.common.voice_state import tool_review_snapshot_from_logs
        from tests.test_staked_tool_close import AXES, EPOCHS
        cut = tool_review_snapshot_from_logs(
            recs, block=100, since_block=0, epochs=EPOCHS, accuracies={},
            axis_ids=AXES, excluded_reviewers=[AUD_B.upper().replace("0X", "0x")])
        assert cut["tools"][DIGEST]["fields"]["correctness"] == {
            "sum": 300 * 90, "mass": 300}
        assert cut["tools"][DIGEST]["fields"]["malware"] == {"sum": 0, "mass": 300}
        assert cut["tools"][DIGEST]["axes"]["evolution"] == {"sum": 300 * 50, "mass": 300}
        assert [r["reviewer"] for r in cut["rounds"][DIGEST][0]["reviews"]] == [
            r["reviewer"] for r in plain["rounds"][DIGEST][0]["reviews"]]


class TestClose:
    POOL = 7.0
    SHARES = {AUD_A: 0.3, AUD_B: 0.2}

    def _close(self, **kw):
        kp_a = Keypair.generate()
        batches = _batches([_registration_event(1)], kp_a)
        kw.setdefault("rep_shares", self.SHARES)
        return federated_epoch_close(
            canonical_order(batches), emission_pool=self.POOL,
            review_snapshot=_snap(), **kw)

    def test_sanctioned_reviewer_earns_nothing_and_nobody_gains(self):
        plain = self._close()
        cut = self._close(sanctioned_households=[AUD_B])
        assert AUD_B not in cut["agent_mint"]
        # The close zeroes a sanctioned household's REP, so as a reviewer it
        # earns nothing to withhold (an author's mint is withheld below).
        assert AUD_B not in cut["withheld_mint"]
        assert cut["total_mint"] <= self.POOL + 1e-9
        assert cut["authoritative_payload"]["sanctioned_households"] == [AUD_B]

    def test_sanctioned_author_is_withheld_reviewers_untouched(self):
        plain = self._close()
        cut = self._close(sanctioned_households=[AUTHOR])
        assert AUTHOR not in cut["agent_mint"]
        assert cut["withheld_mint"] == {AUTHOR: plain["agent_mint"][AUTHOR]}
        for agent in (AUD_A, AUD_B):
            assert cut["agent_mint"][agent] == plain["agent_mint"][agent]

    def test_household_of_a_bound_agent_is_withheld(self):
        owner = {AUD_A: HOUSE, AUD_B: AUD_B}
        cut = self._close(sanctioned_households=[HOUSE], agent_owner_map=owner,
                          rep_shares={HOUSE: 0.3, AUD_B: 0.2})
        assert HOUSE not in cut["agent_mint"]
        assert AUD_B in cut["agent_mint"]

    def test_no_sanctions_changes_nothing_but_the_payload_field(self):
        plain = self._close()
        assert plain["withheld_mint"] == {}
        assert plain["authoritative_payload"]["sanctioned_households"] == []
