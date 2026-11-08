"""Staked reviews at the federated close (docs/multi_jurisdiction.md
decision 10): positions are the ToolBond ratings and successor rating
carry. (The mint paid from concluded rounds is in test_review_mint.py.) Plus the pure ToolBond log fold that
builds the pinned review snapshot, and determinism of all of it under input
reordering.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List

import pytest
from eth_utils import keccak

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import Keypair
from nodes.common.federated_reconcile import (
    REVIEWER_SHARE_BPS_DEFAULT,
    compute_tool_mint,
    federated_epoch_close,
)
from nodes.common.voice_state import tool_review_snapshot_from_logs
from nodes.common.library.charter import DEFAULT_CHARTER
from nodes.common.world_model_substrate.events import ToolUsed
from nodes.common.world_model_substrate.tool_usage import tool_usage_from_events

from tests.test_tool_mint import (
    AUTHOR,
    DIGEST,
    _batches,
    _receipt_event,
    _registration_event,
)

DEP = "cc" * 32
SUCC = "dd" * 32
AUD_A = "0x" + "a1" * 20
AUD_B = "0x" + "b2" * 20
REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}
E18 = 10 ** 18
SUPPLY = 1000 * E18
AXES = list(DEFAULT_CHARTER.ids)
CI = AXES.index("correctness")


def _rev(who: str, stake: int, acc: Any = E18, axes=("correctness",), harm=False):
    return {"reviewer": who, "stake": stake, "accuracy": acc,
            "axes": list(axes), "harm": harm}


def _snapshot(reviews: List[Dict[str, Any]], *, digest: str = DIGEST,
              bps: Any = 4000, prior: Any = None,
              tools: Any = None) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "supply": SUPPLY, "tools": tools or {},
        "rounds": {digest: [{"epoch": 1, "supply": SUPPLY,
                             "prior_mass": prior or {}, "reviews": reviews}]},
    }
    if bps is not None:
        snap["reviewer_share_bps"] = bps
    return snap


def _agent_totals(out: Dict[str, Any]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for bucket in out["node_agent"].values():
        for agent, amount in bucket.items():
            totals[agent] = totals.get(agent, 0.0) + amount
    return totals


def _receipts(n: int = 2) -> List[Dict[str, Any]]:
    return [_receipt_event(i, f"caller-{i}") for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# 2. Positions are the ToolBond ratings
# ---------------------------------------------------------------------------


class TestRatingsArePositions:
    TOOLS = {DIGEST: {"axes": {"correctness": {"sum": 300 * 90 * E18, "mass": 300 * E18},
                               "simplicity": {"sum": -100 * 20 * E18, "mass": 100 * E18}},
                      "harm": {"sum": 400 * 25 * E18, "mass": 400 * E18}}}

    def test_head_mass_and_harm_from_the_snapshot(self):
        out = compute_tool_mint([], registrations=REGS,
                                review_snapshot={"supply": SUPPLY, "tools": self.TOOLS})
        pos = out["positions_next"][DIGEST]
        si = AXES.index("simplicity")
        assert pos["head"][CI] == pytest.approx(0.9)
        assert pos["head"][si] == pytest.approx(-0.2)
        assert pos["mass"][CI] == pytest.approx(0.3)
        assert pos["mass"][si] == pytest.approx(0.1)
        assert pos["harm"] == pytest.approx(0.25)
        assert pos["harm_mass"] == pytest.approx(0.4)
        assert all(pos["head"][i] == 0.0 for i in range(len(AXES)) if i not in (CI, si))

    def test_unrated_tool_sits_at_zero_head_and_mass(self):
        out = compute_tool_mint([], registrations=REGS)
        pos = out["positions_next"][DIGEST]
        assert pos == {"head": [0.0] * len(AXES), "mass": [0.0] * len(AXES),
                       "harm": 0.0, "harm_mass": 0.0}

    def test_gossip_review_axes_move_nothing(self):
        """Axis scores on a tool_used event (an old daemon) no longer drift
        the position: ToolBond is the single source of ratings."""
        ev = _receipt_event(1, "caller-1")
        ev["axes"] = {"correctness": 1.0}
        out = compute_tool_mint([ev], registrations=REGS,
                                rep_shares={"caller-1": 1.0})
        assert out["positions_next"][DIGEST]["head"][CI] == 0.0

    def test_no_credibility_or_review_book_in_the_output(self):
        out = compute_tool_mint(_receipts(), registrations=REGS)
        assert "credibility_next" not in out and "tool_review_book_next" not in out
        kp = Keypair.generate()
        close = federated_epoch_close(canonical_order(
            _batches([_registration_event(1)] + _receipts(), kp)), emission_pool=1.0)
        assert "tool_credibility" not in close and "tool_review_book" not in close

    def test_close_writes_ratings_on_the_world(self):
        kp = Keypair.generate()
        close = federated_epoch_close(
            canonical_order(_batches([_registration_event(1)] + _receipts(), kp)),
            emission_pool=1.0, review_snapshot={"supply": SUPPLY, "tools": self.TOOLS})
        assert close["tool_positions"][DIGEST]["head"][CI] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 3. ToolBond log fold (the pinned snapshot builder)
# ---------------------------------------------------------------------------


D = bytes.fromhex(DIGEST)
KEYS = [keccak(text=a).hex() for a in AXES]


def _rec(block: int, name: str, index: int = 0, **args) -> Dict[str, Any]:
    return {"block": block, "index": index, "event": name, "args": args}


def _scores(**vals) -> bytes:
    out = bytearray()
    for a in AXES:
        v = vals.get(a)
        out.append((-128 if v is None else v) & 0xFF)
    return bytes(out)


def _reviewed(block, who, epoch=1, weight=0, harm=255, stake=0, index=0, **scores):
    return _rec(block, "Reviewed", index=index, digest=D, epoch=epoch, reviewer=who,
                scores=_scores(**scores) if scores else b"", harm=harm, stake=stake,
                weight=weight, evidenceHash=b"\x00" * 32)


def _finalized(block, epoch=1, index=0):
    return _rec(block, "Finalized", index=index, digest=D, epoch=epoch, harm=0,
                bondSlashed=0, lossPool=0, repPool=0)


def _params(block=1, bps=3000):
    return _rec(block, "ParamsSet", minStake=1, revealWindow=1, degradeBps=1000,
                authorSlashBps=10_000, unbondDelay=1, reviewerShareBps=bps,
                harmWeightBps=20_000)


EPOCHS = {1: {"axis_ids": KEYS, "supply": SUPPLY}, 2: {"axis_ids": KEYS, "supply": SUPPLY}}


def _fold(records, block=100, since=0, acc=None):
    return tool_review_snapshot_from_logs(
        records, block=block, since_block=since, epochs=EPOCHS,
        accuracies=acc or {}, axis_ids=AXES)


class TestReviewSnapshotFold:
    def _round1(self):
        return [
            _params(),
            _reviewed(5, AUD_A, weight=300, harm=10, stake=7, index=0, correctness=90),
            _reviewed(5, AUD_B, weight=200, harm=50, stake=0, index=1,
                      correctness=40, simplicity=-60),
            _finalized(9),
        ]

    def test_rating_sums_reviews_of_finalized_rounds(self):
        snap = _fold(self._round1())
        tool = snap["tools"][DIGEST]
        assert snap["reviewer_share_bps"] == 3000
        assert tool["axes"]["correctness"] == {"sum": 300 * 90 + 200 * 40, "mass": 500}
        assert tool["axes"]["simplicity"] == {"sum": -200 * 60, "mass": 200}
        assert tool["harm"] == {"sum": 300 * 10 + 200 * 50, "mass": 500}

    def test_pinned_block_excludes_later_finalize(self):
        snap = _fold(self._round1(), block=8)
        assert snap["tools"] == {}
        assert snap["reviewer_share_bps"] == 3000

    def test_window_rounds_carry_stake_accuracy_and_prior_mass(self):
        recs = self._round1() + [
            _reviewed(20, AUD_A, epoch=2, weight=100, stake=5, correctness=80),
            _finalized(25, epoch=2),
        ]
        acc = {(DIGEST, 2, 0): 9 * E18 // 10}
        snap = _fold(recs, block=30, since=10, acc=acc)
        assert [r["epoch"] for r in snap["rounds"][DIGEST]] == [2]
        rnd = snap["rounds"][DIGEST][0]
        assert rnd["prior_mass"] == {"correctness": 500, "harm": 500, "simplicity": 200}
        assert rnd["reviews"] == [{"reviewer": AUD_A, "stake": 5, "accuracy": 9 * E18 // 10,
                                   "axes": ["correctness"], "harm": False}]
        assert snap["tools"][DIGEST]["axes"]["correctness"]["mass"] == 600

    def test_uninformative_accuracy_reads_as_none(self):
        snap = _fold(self._round1(), since=0, acc={(DIGEST, 1, 0): 2 ** 64 - 1,
                                                   (DIGEST, 1, 1): E18})
        revs = snap["rounds"][DIGEST][0]["reviews"]
        assert revs[0]["accuracy"] is None and revs[1]["accuracy"] == E18

    def test_unknown_axis_key_is_dropped(self):
        epochs = {1: {"axis_ids": [keccak(text="retired").hex()] + KEYS[1:], "supply": 1}}
        recs = [_reviewed(5, AUD_A, weight=10, life_precious=50, correctness=20),
                _finalized(9)]
        snap = tool_review_snapshot_from_logs(recs, block=10, since_block=0, epochs=epochs,
                                              accuracies={}, axis_ids=AXES)
        assert set(snap["tools"][DIGEST]["axes"]) == {"correctness"}

    def test_overruled_round_counts_at_the_daos_harm(self):
        # ToolBond baab631: an overruled round's harm counts at the DAO's
        # value with the same REP mass, so discovery follows the appeal.
        recs = self._round1()[:-1] + [
            _rec(8, "Overruled", digest=D, epoch=1, harm=0), _finalized(9)]
        assert _fold(recs)["tools"][DIGEST]["harm"] == {"sum": 0, "mass": 500}

    def test_settlement_and_epoch_logs_replace_the_views(self):
        # ReviewSettled carries each accuracy and EpochStarted the axis ids
        # (baab631), so no view read is needed.
        recs = [
            _rec(2, "EpochStarted", epoch=1, charterHash=b"\x00" * 32,
                 axisIds=[bytes.fromhex(k.removeprefix("0x")) for k in KEYS],
                 snapshot=1, supply=SUPPLY),
            *self._round1()[1:3],
            _rec(9, "ReviewSettled", index=0, digest=D, epoch=1, reviewer=AUD_A,
                 accuracy=E18, harmAccuracy=E18),
            _rec(9, "ReviewSettled", index=1, digest=D, epoch=1, reviewer=AUD_B,
                 accuracy=2 ** 64 - 1, harmAccuracy=0),
            _finalized(9, index=2),
        ]
        snap = tool_review_snapshot_from_logs(recs, block=10, since_block=0, epochs={},
                                              accuracies={}, axis_ids=AXES)
        revs = snap["rounds"][DIGEST][0]["reviews"]
        assert [r["accuracy"] for r in revs] == [E18, None]
        assert snap["rounds"][DIGEST][0]["supply"] == SUPPLY
        assert snap["tools"][DIGEST]["axes"]["correctness"]["mass"] == 500

    def test_default_bps_without_params_event(self):
        snap = _fold(self._round1()[1:])
        assert snap["reviewer_share_bps"] == REVIEWER_SHARE_BPS_DEFAULT

    def test_order_independent(self):
        recs = self._round1()
        for i, r in enumerate(recs):
            r["index"] = i
        want = _fold(recs)
        rng = random.Random(7)
        for _ in range(5):
            shuffled = recs[:]
            rng.shuffle(shuffled)
            assert _fold(shuffled) == want


# ---------------------------------------------------------------------------
# 5. Successor merit carry
# ---------------------------------------------------------------------------


class TestSuccessorCarry:
    PRED_POS = {DIGEST: {"head": [0.0] * 6, "mass": [0.0] * 6}}
    PRED_RATING = {DIGEST: {"axes": {"correctness": {"sum": 50 * E18, "mass": E18}},
                            "harm": {"sum": 0, "mass": 0}}}

    def _run(self, succ_author: str, tools: Any = None):
        regs = {
            DIGEST: {"trust_class": "pinned", "author": AUTHOR},
            SUCC: {"trust_class": "pinned", "author": succ_author,
                   "version_of": DIGEST},
        }
        return compute_tool_mint(
            [], registrations=regs,
            review_snapshot={"supply": SUPPLY,
                             "tools": copy.deepcopy(tools or self.PRED_RATING)},
        )

    def test_same_author_inherits_rating(self):
        out = self._run(AUTHOR)
        succ = out["positions_next"][SUCC]
        assert succ["head"][CI] == pytest.approx(0.5)
        assert out["positions_next"][DIGEST]["head"][CI] == pytest.approx(0.5)

    def test_successor_with_its_own_rating_keeps_it(self):
        tools = {**self.PRED_RATING,
                 SUCC: {"axes": {"correctness": {"sum": -20 * E18, "mass": E18}},
                        "harm": {"sum": 0, "mass": 0}}}
        out = self._run(AUTHOR, tools)
        assert out["positions_next"][SUCC]["head"][CI] == pytest.approx(-0.2)

    def test_other_author_inherits_nothing(self):
        out = self._run("someone-else")
        succ = out["positions_next"][SUCC]
        assert succ["head"] == [0.0] * 6
        assert succ["mass"] == [0.0] * 6


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_shuffled_inputs_identical_output(self):
        regs = {
            DEP: {"trust_class": "pinned", "author": "dep-author"},
            DIGEST: {"trust_class": "pinned", "author": AUTHOR, "deps": DEP},
            SUCC: {"trust_class": "pinned", "author": AUTHOR, "version_of": DIGEST},
        }
        events = [_receipt_event(1, "c1"), _receipt_event(3, "c3"),
                  _receipt_event(4, "c1", digest=SUCC), _receipt_event(5, "c4", digest=DEP)]
        snap = {"reviewer_share_bps": 2500, "supply": SUPPLY,
                "tools": {DIGEST: {"axes": {"correctness": {"sum": 7 * E18, "mass": E18},
                                            "simplicity": {"sum": 3 * E18, "mass": E18}},
                                   "harm": {"sum": 5, "mass": 1}}},
                "rounds": {DIGEST: [{"epoch": 1, "supply": SUPPLY, "prior_mass": {},
                                     "reviews": [_rev(AUD_A, 7), _rev(AUD_B, 3)]}],
                           DEP: [{"epoch": 1, "supply": SUPPLY, "prior_mass": {},
                                  "reviews": [_rev(AUD_B, 1)]}]}}
        positions = {DIGEST: {"head": [0.0] * 6, "mass": [0.0] * 6}}
        rep = {"c1": 0.2, "c2": 0.1, "c3": 0.05, "c4": 0.3}

        def run(evs, snapshot, pos, shares):
            return compute_tool_mint(evs, registrations=regs,
                                     review_snapshot=snapshot,
                                     rep_shares=shares)

        want = run(events, snap, positions, rep)
        rng = random.Random(11)
        for _ in range(4):
            evs = events[:]
            rng.shuffle(evs)
            snap2 = copy.deepcopy(snap)
            snap2["rounds"] = dict(reversed(list(snap2["rounds"].items())))
            snap2["tools"] = {d: {"axes": dict(reversed(list(e["axes"].items()))),
                                  "harm": e["harm"]}
                              for d, e in snap2["tools"].items()}
            rep2 = dict(reversed(list(rep.items())))
            got = run(evs, snap2, copy.deepcopy(positions), rep2)
            assert got == want

    def test_close_bit_identical_across_batch_order(self):
        kp_a, kp_c = Keypair.generate(), Keypair.generate()
        batches = _batches([_registration_event(1)], kp_a) + _batches(
            [_receipt_event(1, "c1"), _receipt_event(2, "c2")], kp_c)
        snap = _snapshot([_rev(AUD_A, 5), _rev(AUD_B, 2)])
        r1 = federated_epoch_close(canonical_order(batches), emission_pool=3.0,
                                   review_snapshot=snap)
        shuffled = batches[:]
        random.Random(3).shuffle(shuffled)
        r2 = federated_epoch_close(canonical_order(shuffled), emission_pool=3.0,
                                   review_snapshot=copy.deepcopy(snap))
        assert r1["authoritative_payload"] == r2["authoritative_payload"]
        assert r1["tool_positions"] == r2["tool_positions"]


# ---------------------------------------------------------------------------
# 7. Plumbing: attest_tools gossips own_share, never axes; ABI matches ToolBond
# ---------------------------------------------------------------------------


async def test_attest_tools_gossips_nothing(tmp_path):
    """Attestations stay local (Decision 2026-11-07): the row keeps the
    composition estimate and the review, the store has no gossip rail."""
    from atn.agent_tools import execute_tool
    from tests.atn.test_tool_store import (
        _family,
        _make_runtime,
        _register_echo,
    )
    rt = _make_runtime(tmp_path)
    await _family(rt)
    await _register_echo(rt)
    assert not hasattr(rt.tool_store, "event_sink")
    out = await execute_tool(
        "attest_tools",
        {"judgments": [{"tool": "echo_tool", "ok": True, "own_share": 1.4,
                        "axes": {"correctness": 0.5}, "harm": 0.1},
                       {"tool": "echo_tool", "ok": True, "own_share": "junk"}],
         "context": "composing tools"},
        rt, caller_id="parent",
    )
    assert out["attested"] == 2
    rows = list(rt.tool_store._iter_attestations())
    assert rows[0]["own_share"] == 1.0
    assert rows[0]["axes"] == {"correctness": 0.5} and rows[0]["harm"] == 0.1
    assert "own_share" not in rows[1]


def test_tool_bond_event_abi_matches_contract():
    import json
    from pathlib import Path

    from nodes.common.voice_state import _TOOL_BOND_EVENTS_ABI

    art = Path("C:/code/dao/trustless-contracts/artifacts/contracts/"
               "ToolBond.sol/ToolBond.json")
    if not art.exists():
        pytest.skip("trustless-contracts ToolBond artifact not built")
    events = {e["name"]: e for e in json.loads(art.read_text())["abi"]
              if e["type"] == "event"}
    if "Reviewed" not in events:
        pytest.skip("trustless-contracts ToolBond artifact predates 448023a")

    def shape(e):
        return [(i["type"], bool(i["indexed"])) for i in e["inputs"]]

    for spec in _TOOL_BOND_EVENTS_ABI:
        assert shape(spec) == shape(events[spec["name"]]), spec["name"]
