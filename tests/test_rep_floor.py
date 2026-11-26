"""The REP floor (owner decision 2026-11-26).

Registry ``jurisdiction.autonet.rep_floor`` (REP base units, read as value
units: / 1e12) is measured against what the service market produced: the
value of the pool the close splits (new tool share plus carry, at the
pinned parities). When that falls short, the difference is split over all
the weights exactly like the pool, as if the market had produced the floor
in fees: each earning household's claim carries its ``seed``, in its leaf,
and the seed of every part that goes unpaid lapses (never carried, never
given to anyone else). A close where no tool concludes seeds nothing.
These tests pin the close's side; the contract's side (the stored floor,
the per-epoch cap, earnedValue) is in test/index_payments.test.js.
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
    def test_a_pool_below_the_floor_seeds_the_difference_like_the_pool(self):
        out = _close(floor=5_000_000, window={TOKEN: 1_000_000})
        assert out["pool_value"] == 1_000_000
        v = out["total_value"]
        assert 999_997 <= v <= 1_000_000
        seeds = _seeds(out)
        assert set(seeds) == set(FRACTIONS)
        for h, f in FRACTIONS.items():
            # The same fraction of the seed as of the pool.
            assert abs(seeds[h] - 4_000_000 * f) <= 1
            assert abs(seeds[h] * 1_000_000 - out["claims"][h]["tokens"][TOKEN] * 4_000_000) \
                <= 4_000_000
        # Nothing unpaid here, so the floor is met, less the split's dust.
        assert out["seed_lapsed"] == 0
        assert 0 <= 5_000_000 - (v + out["total_seed"]) <= 2 * len(seeds)
        assert out["total_seed"] == sum(seeds.values())
        assert out["rep_floor"] == 5_000_000
        # Each household's tokens and seed sit in one claim.
        assert out["claims"][AUTHOR]["tokens"][TOKEN] > 0

    def test_a_pool_at_or_above_the_floor_seeds_nothing(self):
        for floor in (500_000, 1_000_000):
            out = _close(floor=floor, window={TOKEN: 1_000_000})
            assert _seeds(out) == {} and out["total_seed"] == 0
            assert all(c["seed"] == 0 for c in out["claims"].values())

    def test_the_floor_is_measured_against_the_pool_not_the_claims(self):
        # Correctness 0 leaves the author side (60% of the pool) unpaid: the
        # claims are worth 40% of the pool, but the pool is what the market
        # produced, so a floor at the pool's value seeds nothing.
        out = _close(floor=1_000_000, window={TOKEN: 1_000_000}, snap=_snap(correctness=0.0))
        assert out["total_value"] <= 400_000
        assert out["total_seed"] == 0 and out["seed_lapsed"] == 0

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

    def test_a_withheld_part_lapses_and_no_other_seed_grows(self):
        """The seed of every part that goes unpaid lapses; the others keep
        exactly what they would have had (no redistribution)."""
        base = _seeds(_close(floor=1_000_000, window={}))
        assert base == {R1: 200_000, R2: 200_000, AUTHOR: 600_000}
        cases = {
            # A sanctioned author household: withheld after the split.
            "sanctioned author": (dict(sanctioned_households=[AUTHOR]), {AUTHOR: 0}),
            # A malware verdict: the author side is unpaid.
            "malware": (dict(snap=_snap(malware=0.6)), {AUTHOR: 0}),
            # Correctness 0: the author side is unminted.
            "low ratings": (dict(snap=_snap(correctness=0.0)), {AUTHOR: 0}),
            # Novelty 0.5 without dependencies: the author keeps half its
            # side; the other half (1 - N) pays nobody.
            "1 - N without dependencies": (dict(snap=_snap(novelty=0.5)), {AUTHOR: 300_000}),
        }
        for name, (kw, changed) in cases.items():
            out = _close(floor=1_000_000, window={}, **kw)
            seeds = _seeds(out)
            want = {h: s for h, s in {**base, **changed}.items() if s}
            assert seeds == want, name
            lost = sum(base.values()) - sum(seeds.values())
            assert out["seed_lapsed"] == lost and lost > 0, name
            assert out["tool_rounds"]["pool_carry"] == {}, name

    def test_a_sanctioned_reviewers_seed_lapses(self):
        # R2's reviews weigh nothing and its accuracy cut of the reviewer
        # share goes unpaid: R1 and the author keep the fractions they had.
        base = _seeds(_close(floor=1_000_000, window={}))
        out = _close(floor=1_000_000, window={}, sanctioned_households=[R2])
        seeds = _seeds(out)
        assert R2 not in seeds
        for h in (R1, AUTHOR):
            assert abs(seeds[h] - base[h]) <= 1
        assert abs(out["seed_lapsed"] - base[R2]) <= 1

    def test_a_dependency_that_cannot_be_paid_forfeits_its_seed(self):
        # The author keeps N = 0.5 of its side; its one dependency is not
        # registered, so its 0.5 is forfeited, and so is its seed.
        dep = "cc" * 32
        regs = {DIGEST: {"trust_class": "pinned", "author": AUTHOR, "deps": dep}}
        canonical = canonical_order(_batches([_registration_event(1)], Keypair.generate()))
        out = federated_epoch_close(
            canonical, tool_share_window={}, parity=PARITY, tool_registrations=regs,
            rep_shares=SHARES, review_snapshot=_snap(novelty=0.5), rep_floor=1_000_000)
        assert _seeds(out) == {R1: 200_000, R2: 200_000, AUTHOR: 300_000}
        assert out["seed_lapsed"] == 300_000

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
