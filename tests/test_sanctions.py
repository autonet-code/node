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
    _receipts,
    _registration_event,
    _rev,
    _snapshot,
)
from tests import test_staked_tool_close as stc
from tests.test_tool_mint import AUTHOR

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


class TestReviews:
    def test_sanctioned_reviews_move_no_rating_but_keep_their_round(self):
        recs = stc.TestReviewSnapshotFold()._round1()
        plain = _fold(recs)
        from nodes.common.voice_state import tool_review_snapshot_from_logs
        from tests.test_staked_tool_close import AXES, EPOCHS
        cut = tool_review_snapshot_from_logs(
            recs, block=100, since_block=0, epochs=EPOCHS, accuracies={},
            axis_ids=AXES, excluded_reviewers=[AUD_B.upper().replace("0X", "0x")])
        assert cut["tools"][DIGEST]["axes"]["correctness"] == {
            "sum": 300 * 90, "mass": 300}
        assert "simplicity" not in cut["tools"][DIGEST]["axes"]
        assert cut["tools"][DIGEST]["harm"] == {"sum": 300 * 10, "mass": 300}
        assert [r["reviewer"] for r in cut["rounds"][DIGEST][0]["reviews"]] == [
            r["reviewer"] for r in plain["rounds"][DIGEST][0]["reviews"]]


class TestClose:
    POOL = 7.0

    def _close(self, **kw):
        kp_a, kp_c = Keypair.generate(), Keypair.generate()
        batches = _batches([_registration_event(1)], kp_a) + _batches(
            _receipts(3), kp_c)
        return federated_epoch_close(
            canonical_order(batches), emission_pool=self.POOL,
            review_snapshot=_snapshot([_rev(AUD_A, 2), _rev(AUD_B, 1)]), **kw)

    def test_withheld_mint_stays_unminted(self):
        plain = self._close()
        cut = self._close(sanctioned_households=[AUD_B])
        assert AUD_B not in cut["agent_mint"]
        assert cut["withheld_mint"] == {AUD_B: plain["agent_mint"][AUD_B]}
        # Nobody else gains from the sanction.
        for agent in (AUTHOR, AUD_A):
            assert cut["agent_mint"][agent] == plain["agent_mint"][agent]
        assert cut["total_mint"] == pytest.approx(
            plain["total_mint"] - plain["agent_mint"][AUD_B], abs=1e-6)
        assert cut["authoritative_payload"]["sanctioned_households"] == [AUD_B]

    def test_household_of_a_bound_agent_is_withheld(self):
        cut = self._close(sanctioned_households=[HOUSE],
                          agent_owner_map={AUD_A: HOUSE})
        assert AUD_A not in cut["agent_mint"]
        assert AUD_B in cut["agent_mint"]

    def test_no_sanctions_changes_nothing_but_the_payload_field(self):
        plain = self._close()
        assert plain["withheld_mint"] == {}
        assert plain["authoritative_payload"]["sanctioned_households"] == []
