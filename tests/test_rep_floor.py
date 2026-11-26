"""The REP floor (owner decision 2026-11-26).

Registry ``jurisdiction.autonet.rep_floor`` (REP base units, read as value
units: / 1e12) is the least REP value a close in which tools conclude pays.
When the claims are worth less, the difference is seeded to the earning
households in proportion to their weights: each claim's ``seed``, in its
leaf. A close where no household earns seeds nothing, and nothing carries
over. These tests pin the close's side; the contract's side (the stored
floor, the per-epoch cap, earnedValue) is in test/index_payments.test.js.
"""

from __future__ import annotations

import pytest

from nodes.common.authoritative_encoding import (
    decode_claims_blob,
    encode_authoritative_payload,
    encode_claims_blob,
)
from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import Keypair
from nodes.common.federated_reconcile import federated_epoch_close, seed_split
from nodes.common.mint_merkle import (
    claim_leaves,
    claim_merkle_proof,
    claim_merkle_root,
    verify_claim_proof,
)
from nodes.common.mint_params import REP_FLOOR_KEY, rep_floor_from_writes

from tests.test_tool_mint import DIGEST, _batches, _registration_event, review, tool_cells

E18 = 10 ** 18
AUTHOR = "0x" + "d4" * 20
R1, R2, R3 = ("0x" + c * 20 for c in ("a1", "b2", "c3"))
REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}
SHARES = {R1: 0.3, R2: 0.2, R3: 0.1, AUTHOR: 0.4}
TOKEN = "0x" + "a6" * 20       # 6 decimals, parity 1e30: value == amount
PARITY = {TOKEN: 10 ** 30}
# One tool, reviewed by R1 and R2 at full accuracy and rated at the top:
# its weight 0.5 pays 0.1 to each reviewer and 0.3 to the author, so the
# earners' fractions are 0.2, 0.2 and 0.6 and nothing is left unpaid.
FRACTIONS = {R1: 0.2, R2: 0.2, AUTHOR: 0.6}


def _snap(reviewers=(R1, R2), **cells):
    rounds = ({DIGEST: [{"epoch": 1, "supply": 1000 * E18,
                         "reviews": [review(r) for r in reviewers]}]}
              if reviewers else {})
    return {"supply": 1000 * E18, "reviewer_share_bps": 4000,
            "tools": {DIGEST: tool_cells(**cells)}, "rounds": rounds}


def _close(*, floor, window=None, snap=None, state=None, **kw):
    canonical = canonical_order(_batches([_registration_event(1)], Keypair.generate()))
    return federated_epoch_close(
        canonical, tool_share_window=window, parity=PARITY,
        tool_registrations=REGS, rep_shares=SHARES,
        review_snapshot=snap if snap is not None else _snap(),
        tool_rounds=state, rep_floor=floor, **kw)


def _seeds(out):
    return {h: c["seed"] for h, c in out["claims"].items() if c["seed"]}


class TestTheFloor:
    def test_fees_below_the_floor_seed_the_difference_by_weight(self):
        out = _close(floor=5_000_000, window={TOKEN: 1_000_000})
        v = out["total_value"]
        assert 999_997 <= v <= 1_000_000
        seeds = _seeds(out)
        assert set(seeds) == set(FRACTIONS)
        for h, f in FRACTIONS.items():
            assert abs(seeds[h] - (5_000_000 - v) * f) <= 1
        # The floor is met, less the split's dust.
        assert 0 <= 5_000_000 - (v + out["total_seed"]) <= len(seeds)
        assert out["total_seed"] == sum(seeds.values())
        assert out["rep_floor"] == 5_000_000
        # Each household's tokens and seed sit in one claim.
        assert out["claims"][AUTHOR]["tokens"][TOKEN] > 0

    def test_fees_at_or_above_the_floor_seed_nothing(self):
        out = _close(floor=500_000, window={TOKEN: 1_000_000})
        assert out["total_value"] > 500_000
        assert _seeds(out) == {} and out["total_seed"] == 0
        assert all(c["seed"] == 0 for c in out["claims"].values())

    def test_no_floor_seeds_nothing(self):
        out = _close(floor=0, window={})
        assert out["claims"] == {} and out["total_seed"] == 0

    def test_empty_pools_with_a_conclusion_seed_the_whole_floor_by_weight(self):
        for window in ({}, None):
            out = _close(floor=1_000_000, window=window)
            assert out["total_value"] == 0
            assert _seeds(out) == {h: int(1_000_000 * f) for h, f in FRACTIONS.items()}
            assert all(c["tokens"] == {} for c in out["claims"].values())
            # A seed-only claim still has a leaf, proven like any other.
            leaves = claim_leaves(out["claims"])
            assert sorted(l[0].lower() for l in leaves) == sorted(FRACTIONS)
            root = claim_merkle_root(out["claims"], concluded=out["concluded"])
            for earner, tokens, amounts, seed in leaves:
                proof = claim_merkle_proof(out["claims"], earner, concluded=out["concluded"])
                assert tokens == [] and amounts == [] and seed > 0
                assert verify_claim_proof(root, earner, tokens, amounts, seed, proof)

    def test_no_conclusion_lapses_the_floor_and_nothing_carries(self):
        quiet = _close(floor=1_000_000, window={}, snap=_snap(reviewers=()))
        assert quiet["claims"] == {} and quiet["total_seed"] == 0
        assert quiet["tool_rounds"]["pool_carry"] == {}
        # The next close with a conclusion seeds one floor, not two.
        nxt = _close(floor=1_000_000, window={}, state=quiet["tool_rounds"])
        assert 1_000_000 - len(FRACTIONS) <= nxt["total_seed"] <= 1_000_000

    def test_excluded_earners_take_no_seed(self):
        # R2 sanctioned: withheld, and its seed with it. A malware verdict
        # leaves the author side unpaid: no seed for the author either. The
        # difference goes to the earners that remain (R1).
        out = _close(floor=1_000_000, window={}, sanctioned_households=[R2])
        assert R2 not in _seeds(out)
        mal = _close(floor=1_000_000, window={}, snap=_snap(malware=0.6))
        assert AUTHOR not in _seeds(mal)
        assert set(_seeds(mal)) == {R1, R2}

    def test_unpaid_weight_is_not_an_earner(self):
        # Correctness 0: the author side is unpaid (unminted), so only the
        # reviewers earn, and they take the whole difference by weight.
        out = _close(floor=1_000_000, window={TOKEN: 1_000_000}, snap=_snap(correctness=0.0))
        assert AUTHOR not in out["claims"]
        seeds = _seeds(out)
        assert set(seeds) == {R1, R2}
        assert abs(seeds[R1] - seeds[R2]) <= 1
        assert 0 <= 1_000_000 - out["total_value"] - out["total_seed"] <= 2

    def test_deterministic(self):
        a = _close(floor=3_333_333, window={TOKEN: 777_777})
        b = _close(floor=3_333_333, window={TOKEN: 777_777})
        assert a["claims"] == b["claims"] and a["total_seed"] == b["total_seed"]


class TestEncoding:
    def test_schema_10_commits_the_seed_the_floor_and_the_total(self):
        import json
        out = _close(floor=2_000_000, window={TOKEN: 1_000_000})
        raw = encode_authoritative_payload(
            out["authoritative_payload"], epoch_id="e1",
            epoch_root_hex="11" * 32, prev_epoch_root_hex="00" * 32)
        d = json.loads(raw)
        assert d["schema"] == 10
        assert d["rep_floor"] == "2000000"
        assert d["total_seed"] == str(out["total_seed"])
        assert list(d)[7:10] == ["total_value", "rep_floor", "total_seed"]
        assert d["claims"][AUTHOR]["seed"] == str(out["claims"][AUTHOR]["seed"])

    def test_the_claims_blob_round_trips_the_seed(self):
        out = _close(floor=1_000_000, window={})
        claims = decode_claims_blob(encode_claims_blob(out["claims"], concluded=out["concluded"]))
        assert {h: c["seed"] for h, c in claims.items()} == _seeds(out)
        assert claim_merkle_root(claims, concluded=out["concluded"]) == \
            claim_merkle_root(out["claims"], concluded=out["concluded"])


class TestSplitAndParams:
    def test_seed_split_is_integer_and_the_dust_lapses(self):
        assert seed_split(10, {"a": 1, "b": 1, "c": 1}) == {"a": 3, "b": 3, "c": 3}
        assert seed_split(10, {"a": 3, "b": 1}) == {"a": 7, "b": 2}
        assert seed_split(1, {"a": 1, "b": 1}) == {}
        assert seed_split(0, {"a": 1}) == {} and seed_split(-5, {"a": 1}) == {}
        assert seed_split(10, {}) == {}

    @pytest.mark.parametrize("raw,want", [
        (None, 0), ("", 0), ("5000000000000000000", 5_000_000),
        ("5000000000000000999", 5_000_000), ("999999999999", 0),
        ("not a number", 0), ("-5", 0), ("1e18", 0), (" 5", 0), ("5.0", 0),
        ("9" * 77, 10 ** 65 - 1), ("9" * 78, 0), ("٥", 0),
    ])
    def test_rep_floor_is_read_as_substrate_reads_it(self, raw, want):
        writes = {} if raw is None else {REP_FLOOR_KEY: raw}
        assert rep_floor_from_writes(writes) == want

    def test_the_key_is_substrates(self):
        assert REP_FLOOR_KEY == "jurisdiction.autonet.rep_floor"
