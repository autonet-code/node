"""Reviews at the federated close: the carried ratings are ToolBond's plain
means (Decision 2026-11-09) with successor rating carry. (The mint paid from
concluded rounds is in test_review_mint.py.) Plus the pure ToolBond log fold
that builds the pinned review snapshot, and determinism of all of it under
input reordering.
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

from tests.test_tool_mint import (
    AUTHOR,
    DIGEST,
    _batches,
    _receipt_event,
    _registration_event,
    review,
    tool_cells,
)

DEP = "cc" * 32
SUCC = "dd" * 32
AUD_A = "0x" + "a1" * 20
AUD_B = "0x" + "b2" * 20
REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}
E18 = 10 ** 18
SUPPLY = 1000 * E18
AXES = list(DEFAULT_CHARTER.ids)


def _rev(who: str, acc: Any = E18) -> Dict[str, Any]:
    return review(who, acc)


def _snapshot(reviews: List[Dict[str, Any]], *, digest: str = DIGEST,
              bps: Any = 4000, tools: Any = None) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "supply": SUPPLY, "tools": tools or {},
        "rounds": {digest: [{"epoch": 1, "supply": SUPPLY, "reviews": reviews}]},
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
# 2. The carried ratings are the ToolBond means
# ---------------------------------------------------------------------------


class TestRatings:
    TOOLS = {DIGEST: tool_cells(correctness=0.9, simplicity=0.2, novelty=0.4,
                                malware=0.25, value=-0.2, mass=300 * E18)}

    def test_means_and_mass_from_the_snapshot(self):
        out = compute_tool_mint([], registrations=REGS,
                                review_snapshot={"supply": SUPPLY, "tools": self.TOOLS})
        r = out["ratings_next"][DIGEST]
        assert r["correctness"] == pytest.approx(0.9)
        assert r["simplicity"] == pytest.approx(0.2)
        assert r["novelty"] == pytest.approx(0.4)
        assert r["malware"] == pytest.approx(0.25)
        assert r["value"] == pytest.approx(-0.2)
        assert r["axes"][AXES[0]] == pytest.approx(-0.2)
        assert r["mass"] == pytest.approx(0.3)

    def test_unrated_tool_has_no_means(self):
        out = compute_tool_mint([], registrations=REGS)
        r = out["ratings_next"][DIGEST]
        assert r["correctness"] is None and r["axes"] == {} and r["mass"] == 0.0

    def test_gossip_review_axes_move_nothing(self):
        """Axis scores on a tool_used event (an old daemon) rate nothing:
        ToolBond is the single source of ratings."""
        ev = _receipt_event(1, "caller-1")
        ev["axes"] = {AXES[0]: 1.0}
        out = compute_tool_mint([ev], registrations=REGS,
                                rep_shares={"caller-1": 1.0})
        assert out["ratings_next"][DIGEST]["axes"] == {}

    def test_no_credibility_or_review_book_in_the_output(self):
        out = compute_tool_mint(_receipts(), registrations=REGS)
        assert "credibility_next" not in out and "tool_review_book_next" not in out
        kp = Keypair.generate()
        close = federated_epoch_close(canonical_order(
            _batches([_registration_event(1)] + _receipts(), kp)), emission_pool=1.0)
        assert "tool_credibility" not in close and "tool_review_book" not in close
        assert "tool_positions" not in close

    def test_close_returns_the_ratings(self):
        kp = Keypair.generate()
        close = federated_epoch_close(
            canonical_order(_batches([_registration_event(1)] + _receipts(), kp)),
            emission_pool=1.0, review_snapshot={"supply": SUPPLY, "tools": self.TOOLS})
        assert close["tool_ratings"][DIGEST]["correctness"] == pytest.approx(0.9)


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


def _reviewed(block, who, epoch=1, weight=0, index=0, *, c=50, s=50, n=50, m=0,
              **scores):
    return _rec(block, "Reviewed", index=index, digest=D, epoch=epoch, reviewer=who,
                correctness=c, simplicity=s, novelty=n, malware=m,
                scores=_scores(**scores) if scores else b"",
                weight=weight, evidenceHash=b"\x00" * 32)


def _finalized(block, epoch=1, index=0):
    return _rec(block, "Finalized", index=index, digest=D, epoch=epoch, repPool=0)


def _params(block=1, bps=3000):
    return _rec(block, "ParamsSet", minStake=1, revealWindow=1, degradeBps=1000,
                authorSlashBps=10_000, unbondDelay=1, reviewerShareBps=bps)


EPOCHS = {1: {"axis_ids": KEYS, "supply": SUPPLY, "unscored_bps": 5000},
          2: {"axis_ids": KEYS, "supply": SUPPLY, "unscored_bps": 5000}}
U = 50     # an unscored axis at the default 5000 bps, in score points


def _fold(records, block=100, since=0, acc=None):
    return tool_review_snapshot_from_logs(
        records, block=block, since_block=since, epochs=EPOCHS,
        accuracies=acc or {}, axis_ids=AXES)


class TestReviewSnapshotFold:
    def _round1(self):
        return [
            _params(),
            _reviewed(5, AUD_A, weight=300, index=0, c=90, s=20, n=70, m=0,
                      life_precious=90),
            _reviewed(5, AUD_B, weight=200, index=1, c=40, s=60, n=10, m=1,
                      life_precious=40, evolution=-60),
            _finalized(9),
        ]

    def test_sums_every_field_and_axis_of_finalized_rounds(self):
        snap = _fold(self._round1())
        tool = snap["tools"][DIGEST]
        assert snap["reviewer_share_bps"] == 3000
        f = tool["fields"]
        assert f["correctness"] == {"sum": 300 * 90 + 200 * 40, "mass": 500}
        assert f["simplicity"] == {"sum": 300 * 20 + 200 * 60, "mass": 500}
        assert f["novelty"] == {"sum": 300 * 70 + 200 * 10, "mass": 500}
        # A malware yes counts 100: sum / mass / 100 is the yes share.
        assert f["malware"] == {"sum": 200 * 100, "mass": 500}
        ax = tool["axes"]
        assert ax["life_precious"] == {"sum": 300 * 90 + 200 * 40, "mass": 500}
        # Unscored axes count at the default with the review's weight.
        assert ax["evolution"] == {"sum": 300 * U - 200 * 60, "mass": 500}
        assert ax["self_preservation"] == {"sum": 500 * U, "mass": 500}

    def test_empty_scores_count_every_axis_at_the_default(self):
        recs = [_reviewed(5, AUD_A, weight=7), _finalized(9)]
        ax = _fold(recs)["tools"][DIGEST]["axes"]
        assert set(ax) == set(AXES)
        assert all(cell == {"sum": 7 * U, "mass": 7} for cell in ax.values())

    def test_the_epochs_unscored_default_and_truncation(self):
        epochs = {1: {"axis_ids": KEYS, "supply": SUPPLY, "unscored_bps": 3333}}
        recs = [_reviewed(5, AUD_A, weight=7), _finalized(9)]
        snap = tool_review_snapshot_from_logs(recs, block=10, since_block=0,
                                              epochs=epochs, accuracies={},
                                              axis_ids=AXES)
        # ToolBond adds w x 3333 / 100, truncated: 233.31 -> 233.
        assert snap["tools"][DIGEST]["axes"]["evolution"] == {"sum": 233, "mass": 7}

    def test_pinned_block_excludes_later_finalize(self):
        snap = _fold(self._round1(), block=8)
        assert snap["tools"] == {}
        assert snap["reviewer_share_bps"] == 3000

    def test_window_rounds_carry_accuracy_and_their_own_cells(self):
        recs = self._round1() + [
            _reviewed(20, AUD_A, epoch=2, weight=100, c=80),
            _finalized(25, epoch=2),
        ]
        acc = {(DIGEST, 2, 0): 9 * E18 // 10}
        snap = _fold(recs, block=30, since=10, acc=acc)
        assert [r["epoch"] for r in snap["rounds"][DIGEST]] == [2]
        rnd = snap["rounds"][DIGEST][0]
        assert rnd["reviews"] == [{"reviewer": AUD_A, "accuracy": 9 * E18 // 10}]
        assert rnd["fields"]["correctness"] == {"sum": 100 * 80, "mass": 100}
        assert snap["tools"][DIGEST]["fields"]["correctness"]["mass"] == 600

    def test_an_unweighted_review_leaves_the_sums(self):
        """One weight per household: ToolBond takes a review's weight back
        out when an earlier commit of the household reveals later."""
        recs = [_reviewed(5, AUD_A, weight=300, c=90),
                _rec(6, "ReviewUnweighted", digest=D, epoch=1, reviewer=AUD_A),
                _reviewed(6, AUD_B, weight=300, index=1, c=10),
                _finalized(9)]
        f = _fold(recs)["tools"][DIGEST]["fields"]
        assert f["correctness"] == {"sum": 300 * 10, "mass": 300}

    def test_uninformative_accuracy_reads_as_none(self):
        snap = _fold(self._round1(), since=0, acc={(DIGEST, 1, 0): 2 ** 64 - 1,
                                                   (DIGEST, 1, 1): E18})
        revs = snap["rounds"][DIGEST][0]["reviews"]
        assert revs[0]["accuracy"] is None and revs[1]["accuracy"] == E18

    def test_unknown_axis_key_is_dropped(self):
        epochs = {1: {"axis_ids": [keccak(text="retired").hex()] + KEYS[1:],
                      "supply": 1, "unscored_bps": 5000}}
        recs = [_reviewed(5, AUD_A, weight=10, life_precious=50, evolution=20),
                _finalized(9)]
        snap = tool_review_snapshot_from_logs(recs, block=10, since_block=0, epochs=epochs,
                                              accuracies={}, axis_ids=AXES)
        assert "life_precious" not in snap["tools"][DIGEST]["axes"]
        assert snap["tools"][DIGEST]["axes"]["evolution"] == {"sum": 200, "mass": 10}

    def test_verdict_events_fold_per_tool(self):
        recs = self._round1() + [
            _rec(20, "Concluded", digest=D, anchorIndex=3, malwareShare=6 * E18 // 10,
                 appealEnds=99),
            _rec(21, "MalwareRuled", digest=D, malware=False),
            _rec(22, "MalwareSettled", digest=D, malware=False, bondSlashed=0,
                 repBurned=0)]
        assert _fold(recs)["verdicts"][DIGEST] == {
            "concluded": True, "ruling": False, "settled": False}
        assert _fold(recs, block=20)["verdicts"][DIGEST] == {
            "concluded": True, "ruling": None, "settled": None}

    def test_settlement_and_epoch_logs_replace_the_views(self):
        # ReviewSettled carries each accuracy and EpochStarted the axis ids
        # and the unscored default, so no view read is needed.
        recs = [
            _rec(2, "EpochStarted", epoch=1, charterHash=b"\x00" * 32,
                 axisIds=[bytes.fromhex(k.removeprefix("0x")) for k in KEYS],
                 snapshot=1, supply=SUPPLY, unscoredBps=5000),
            *self._round1()[1:3],
            _rec(9, "ReviewSettled", index=0, digest=D, epoch=1, reviewer=AUD_A,
                 accuracy=E18),
            _rec(9, "ReviewSettled", index=1, digest=D, epoch=1, reviewer=AUD_B,
                 accuracy=2 ** 64 - 1),
            _finalized(9, index=2),
        ]
        snap = tool_review_snapshot_from_logs(recs, block=10, since_block=0, epochs={},
                                              accuracies={}, axis_ids=AXES)
        revs = snap["rounds"][DIGEST][0]["reviews"]
        assert [r["accuracy"] for r in revs] == [E18, None]
        assert snap["rounds"][DIGEST][0]["supply"] == SUPPLY
        assert snap["tools"][DIGEST]["fields"]["correctness"]["mass"] == 500

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

    def test_the_close_reads_the_fold(self):
        """End to end: the fold's cells give the close's rating."""
        from nodes.common.federated_reconcile import tool_rating
        r = tool_rating(_fold(self._round1())["tools"][DIGEST], AXES, 1000)
        assert r["correctness"] == pytest.approx((300 * 90 + 200 * 40) / 500 / 100)
        assert r["malware"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 5. Successor merit carry
# ---------------------------------------------------------------------------


class TestSuccessorCarry:
    PRED_RATING = {DIGEST: tool_cells(correctness=0.5)}

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
        assert out["ratings_next"][SUCC]["correctness"] == pytest.approx(0.5)
        assert out["ratings_next"][DIGEST]["correctness"] == pytest.approx(0.5)

    def test_successor_with_its_own_rating_keeps_it(self):
        tools = {**self.PRED_RATING, SUCC: tool_cells(correctness=-0.2)}
        out = self._run(AUTHOR, tools)
        assert out["ratings_next"][SUCC]["correctness"] == pytest.approx(-0.2)

    def test_other_author_inherits_nothing(self):
        out = self._run("someone-else")
        succ = out["ratings_next"][SUCC]
        assert succ["correctness"] is None and succ["mass"] == 0.0


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
                "tools": {DIGEST: tool_cells(correctness=0.07, novelty=0.3),
                          DEP: tool_cells(malware=0.2)},
                "rounds": {DIGEST: [{"epoch": 1, "supply": SUPPLY,
                                     "reviews": [_rev(AUD_A, 7), _rev(AUD_B, 3)]}],
                           DEP: [{"epoch": 1, "supply": SUPPLY,
                                  "reviews": [_rev(AUD_B, 1)]}]}}
        rep = {"c1": 0.2, "c2": 0.1, "c3": 0.05, "c4": 0.3,
               AUD_A: 0.1, AUD_B: 0.2}

        def run(evs, snapshot, shares):
            return compute_tool_mint(evs, registrations=regs,
                                     review_snapshot=snapshot,
                                     rep_shares=shares)

        want = run(events, snap, rep)
        rng = random.Random(11)
        for _ in range(4):
            evs = events[:]
            rng.shuffle(evs)
            snap2 = copy.deepcopy(snap)
            snap2["rounds"] = dict(reversed(list(snap2["rounds"].items())))
            snap2["tools"] = {d: {"axes": dict(reversed(list(e["axes"].items()))),
                                  "fields": dict(reversed(list(e["fields"].items())))}
                              for d, e in reversed(list(snap2["tools"].items()))}
            rep2 = dict(reversed(list(rep.items())))
            got = run(evs, snap2, rep2)
            assert got == want

    def test_close_bit_identical_across_batch_order(self):
        kp_a, kp_c = Keypair.generate(), Keypair.generate()
        batches = _batches([_registration_event(1)], kp_a) + _batches(
            [_receipt_event(1, "c1"), _receipt_event(2, "c2")], kp_c)
        snap = _snapshot([_rev(AUD_A, 5), _rev(AUD_B, 2)],
                         tools={DIGEST: tool_cells(novelty=0.4)})
        r1 = federated_epoch_close(canonical_order(batches), emission_pool=3.0,
                                   review_snapshot=snap)
        shuffled = batches[:]
        random.Random(3).shuffle(shuffled)
        r2 = federated_epoch_close(canonical_order(shuffled), emission_pool=3.0,
                                   review_snapshot=copy.deepcopy(snap))
        assert r1["authoritative_payload"] == r2["authoritative_payload"]
        assert r1["tool_ratings"] == r2["tool_ratings"]


# ---------------------------------------------------------------------------
# 7. Plumbing: attest_tools gossips nothing; the ABI matches ToolBond
# ---------------------------------------------------------------------------


async def test_attest_tools_gossips_nothing(tmp_path):
    """Reviews stay local until ToolBond (Decision 2026-11-07): the row
    keeps the review, the store has no gossip rail, and there is no
    composition estimate (novelty replaced it, Decision 2026-11-09)."""
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
        {"judgments": [{"tool": "echo_tool", "correctness": 0.5, "simplicity": 1,
                        "novelty": 0.2, "malware": False, "own_share": 1.4,
                        "axes": {AXES[0]: 0.5}}],
         "context": "composing tools"},
        rt, caller_id="parent",
    )
    assert out["attested"] == 1
    (row,) = list(rt.tool_store._iter_attestations())
    assert "own_share" not in row
    assert row["axes"] == {AXES[0]: 0.5} and row["novelty"] == 0.2


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
