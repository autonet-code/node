"""Review batches (Decision 2026-11-09): the fixed fields and charter score
encoding, the commitment ToolBond checks, the per-epoch batch store, and
what makes a review valid."""
from __future__ import annotations

import pytest

from atn.runtime.review_batch import (
    NO_SCORE, REVIEW_INPUT_TYPE, EpochBatch, ReviewBatchStore, ReviewEntry,
    commitment_of, encode_scores, normalize_review, review_error,
)

AXES = ["life_precious", "self_preservation", "promotion_of_intelligence",
        "evolution"]
D1 = "aa" * 32
D2 = "bb" * 32
AGENT = "0x" + "12" * 20
EVIDENCE = "ee" * 32


def _entry(digest=D1, **kw):
    base = {"correctness": 0.5, "simplicity": 0.5, "novelty": 0.5, "malware": False}
    base.update(kw)
    return ReviewEntry(digest=digest, **base)


def test_scores_one_int8_per_axis_in_charter_order():
    b = encode_scores({"evolution": 0.5, "life_precious": -1.0}, AXES)
    assert len(b) == 4
    vals = [x - 256 if x > 127 else x for x in b]
    assert vals == [-100, NO_SCORE, NO_SCORE, 50]


def test_scores_clamped_and_empty_when_nothing_scored():
    vals = [x - 256 if x > 127 else x for x in encode_scores({"evolution": 3.0}, AXES)]
    assert vals[3] == 100
    assert encode_scores({}, AXES) == b""
    assert encode_scores({"unknown_axis": 1.0}, AXES) == b""


def test_too_many_axes_rejected():
    with pytest.raises(ValueError):
        encode_scores({"a": 1.0}, [f"a{i}" for i in range(17)])


def test_input_carries_the_fixed_fields_in_hundredths():
    e = _entry(correctness=-0.804, simplicity=1.7, novelty=0.333, malware=True,
               axes={"evolution": 0.2}, evidence_hash=EVIDENCE)
    digest, c, s, n, m, scores, ev = e.to_input(AXES)
    assert (c, s, n, m) == (-80, 100, 33, 1)
    assert digest == bytes.fromhex(D1) and ev == bytes.fromhex(EVIDENCE)
    assert [x - 256 if x > 127 else x for x in scores] == [NO_SCORE] * 3 + [20]


def test_commitment_matches_abi_encoding():
    from eth_abi import encode
    from eth_utils import keccak, to_checksum_address
    inputs = [_entry(axes={"evolution": 1.0}).to_input(AXES)]
    salt = b"\x07" * 32
    assert REVIEW_INPUT_TYPE == "(bytes32,int8,uint8,uint8,uint8,bytes,bytes32)"
    want = keccak(encode(["uint256", "address", REVIEW_INPUT_TYPE + "[]", "bytes32"],
                         [3, to_checksum_address(AGENT), inputs, salt]))
    assert commitment_of(3, AGENT, inputs, salt) == want
    # Any change to the hidden content changes the commitment.
    other = [_entry(axes={"evolution": 0.9}).to_input(AXES)]
    assert commitment_of(3, AGENT, other, salt) != want


def test_commitment_matches_the_contract_abi():
    """The tuple type is the one ToolBond's commitmentOf takes."""
    import json
    from pathlib import Path
    art = Path("C:/code/dao/trustless-contracts/artifacts/contracts/"
               "ToolBond.sol/ToolBond.json")
    if not art.exists():
        pytest.skip("trustless-contracts ToolBond artifact not built")
    fn = next(e for e in json.loads(art.read_text())["abi"]
              if e.get("name") == "commitmentOf")
    comps = fn["inputs"][2]["components"]
    assert "(" + ",".join(c["type"] for c in comps) + ")" == REVIEW_INPUT_TYPE


class TestValidity:
    BASE = {"digest": D1, "correctness": 0.4, "simplicity": 0.2,
            "novelty": 0.9, "malware": False}

    def test_all_four_fields_required(self):
        assert review_error(self.BASE) == ""
        for name in ("correctness", "simplicity", "novelty", "malware"):
            raw = {k: v for k, v in self.BASE.items() if k != name}
            assert name in review_error(raw)
            assert normalize_review(raw, AXES) is None

    def test_malware_must_be_a_boolean(self):
        assert "malware" in review_error({**self.BASE, "malware": 1})

    def test_a_malware_flag_needs_evidence(self):
        flagged = {**self.BASE, "malware": True}
        assert "evidence" in review_error(flagged)
        assert "evidence" in review_error({**flagged, "evidence_hash": "00" * 32})
        assert review_error({**flagged, "evidence_hash": EVIDENCE}) == ""

    def test_normalized_entry_clamps_and_drops_unknown_axes(self):
        e = normalize_review({**self.BASE, "correctness": 3, "novelty": -1,
                              "axes": {"evolution": 0.5, "kindness": 1.0}}, AXES)
        assert (e.correctness, e.novelty) == (1.0, 0.0)
        assert e.axes == {"evolution": 0.5}
        assert e.evidence_hash == "00" * 32


def test_store_queues_seals_and_rolls_to_next_epoch(tmp_path):
    store = ReviewBatchStore(tmp_path, "31337:0xdao")
    store.add(AGENT, 4, AXES, _entry(D1, correctness=1.0))
    store.add(AGENT, 4, AXES, _entry(D2, malware=True, evidence_hash=EVIDENCE))
    # A second review of the same tool replaces the first.
    store.add(AGENT, 4, AXES, _entry(D1, correctness=0.2))
    b = store.load(AGENT, 4, AXES)
    assert [r.digest for r in b.reviews] == [D2, D1]
    assert b.reviews[1].correctness == 0.2 and b.reviews[0].malware is True

    # Once committed, the batch is sealed: new reviews go to the next epoch.
    b.salt, b.commitment = "ab" * 32, "cd" * 32
    store.save(b)
    nxt = store.add(AGENT, 4, AXES, _entry(D1))
    assert nxt.epoch == 5
    assert [x.epoch for x in store.unrevealed()] == [4]

    sealed = store.load(AGENT, 4, AXES).sealed_reveal()
    assert sealed["salt"] == "ab" * 32 and len(sealed["reviews"]) == 2
    assert "stake" not in sealed["reviews"][0]


def test_an_old_entry_without_the_fields_is_dropped_on_load(tmp_path):
    import json
    store = ReviewBatchStore(tmp_path, "j")
    store.add(AGENT, 1, AXES, _entry(D1))
    path = store._path(AGENT, 1)
    data = json.loads(path.read_text())
    data["reviews"].append({"digest": D2, "axes": {}, "harm": 0.5, "stake": 1,
                            "evidence_hash": "00" * 32})
    path.write_text(json.dumps(data))
    assert [r.digest for r in store.load(AGENT, 1, AXES).reviews] == [D1]


def test_batches_are_kept_per_jurisdiction(tmp_path):
    a = ReviewBatchStore(tmp_path, "31337:0xa")
    b = ReviewBatchStore(tmp_path, "31337:0xb")
    a.add(AGENT, 1, AXES, _entry(D1))
    assert b.load(AGENT, 1, AXES).reviews == []
    assert isinstance(a.load(AGENT, 1, AXES), EpochBatch)
