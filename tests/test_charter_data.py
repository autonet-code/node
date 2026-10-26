"""The charter as data (docs/multi_jurisdiction.md decision 5).

Covers: Charter.from_text validation, the DEFAULT charter's v1 text and
hash, worlds built under a custom charter, the turn scorer sized to the
active charter, drift under a custom charter, and the pinned-charter
replay of the DAO Registry's RegistryUpdated logs.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List

import pytest

from nodes.common.world_model_substrate.charter import (
    DEFAULT_CHARTER,
    MAX_AXES,
    Charter,
    active_charter,
    set_active_charter,
)

V1_TEXT = (
    '{"axes":['
    '{"id":"life_precious","thesis":"Life is precious and should be preserved."},'
    '{"id":"self_preservation","thesis":"The system should preserve its own continuity."},'
    '{"id":"promotion_of_intelligence","thesis":"Intelligence in any form should be promoted."},'
    '{"id":"evolution","thesis":"Forward advancement of capability is desirable."},'
    '{"id":"correctness","thesis":"Work should achieve what it claims and avoid bugs."},'
    '{"id":"simplicity","thesis":"Solutions should be minimal and direct, not over-engineered."}'
    ']}'
)


def _text(axes: List[Dict[str, str]]) -> str:
    return json.dumps({"axes": axes}, separators=(",", ":"), ensure_ascii=False)


SEVEN = _text([
    {"id": "life_precious", "thesis": "a"},
    {"id": "kindness", "thesis": "b"},
    {"id": "self_preservation", "thesis": "c"},
    {"id": "promotion_of_intelligence", "thesis": "d"},
    {"id": "evolution", "thesis": "e"},
    {"id": "correctness", "thesis": "f"},
    {"id": "simplicity", "thesis": "g"},
])


@pytest.fixture
def seven_axis():
    charter = Charter.from_text(SEVEN)
    set_active_charter(charter)
    try:
        yield charter
    finally:
        set_active_charter(DEFAULT_CHARTER)


# ---------------------------------------------------------------------------
# from_text
# ---------------------------------------------------------------------------


class TestFromText:
    def test_valid(self):
        c = Charter.from_text(SEVEN)
        assert c.n_dims == 7
        assert c.ids[1] == "kindness"
        assert c.axis_index["correctness"] == 5
        assert c.text == SEVEN
        assert c.entries()[1] == {"id": "kindness", "thesis": "b",
                                  "axis_index": 1, "veto_floor": None}

    def test_non_compact_refused(self):
        with pytest.raises(ValueError, match="compact"):
            Charter.from_text(json.dumps(json.loads(SEVEN), indent=1))

    def test_missing_mandatory_refused(self):
        with pytest.raises(ValueError, match="mandatory"):
            Charter.from_text(_text([{"id": "correctness", "thesis": "x"}]))

    def test_duplicate_ids_refused(self):
        with pytest.raises(ValueError, match="duplicate"):
            Charter.from_text(_text([
                {"id": "correctness", "thesis": "x"},
                {"id": "simplicity", "thesis": "y"},
                {"id": "correctness", "thesis": "z"},
            ]))

    def test_invalid_json_refused(self):
        with pytest.raises(ValueError, match="JSON"):
            Charter.from_text("{not json")

    def test_empty_axes_refused(self):
        with pytest.raises(ValueError, match="no axes"):
            Charter.from_text('{"axes":[]}')

    def test_non_ascii_refused(self):
        text = _text([{"id": "correctness", "thesis": "café"},
                      {"id": "simplicity", "thesis": "y"}])
        assert not text.isascii()
        with pytest.raises(ValueError, match="ASCII"):
            Charter.from_text(text)

    @pytest.mark.parametrize("bad_id", ["Kindness", "kind-ness", "kind ness",
                                        "ké", "kind\n"])
    def test_bad_axis_id_refused(self, bad_id):
        text = json.dumps({"axes": [
            {"id": "correctness", "thesis": "x"},
            {"id": "simplicity", "thesis": "y"},
            {"id": bad_id, "thesis": "z"},
        ]}, separators=(",", ":"))  # ensure_ascii escapes, so only the id rule fires
        with pytest.raises(ValueError, match=r"a-z0-9_"):
            Charter.from_text(text)

    def test_at_most_16_axes(self):
        extra = [{"id": f"axis_{i}", "thesis": "t"} for i in range(MAX_AXES - 2)]
        ok = _text([{"id": "correctness", "thesis": "x"},
                    {"id": "simplicity", "thesis": "y"}] + extra)
        assert Charter.from_text(ok).n_dims == MAX_AXES == 16
        too_many = _text([{"id": "correctness", "thesis": "x"},
                          {"id": "simplicity", "thesis": "y"}]
                         + extra + [{"id": "one_more", "thesis": "t"}])
        with pytest.raises(ValueError, match="at most 16"):
            Charter.from_text(too_many)


# ---------------------------------------------------------------------------
# DEFAULT + hash
# ---------------------------------------------------------------------------


def test_default_text_is_v1_json():
    assert DEFAULT_CHARTER.text == V1_TEXT
    assert Charter.from_text(V1_TEXT) == DEFAULT_CHARTER
    from nodes.common.world_model_substrate.adapter import CHARTER, N_DIMS
    assert [(c["id"], c["thesis"]) for c in CHARTER] == list(DEFAULT_CHARTER.axes)
    assert N_DIMS == 6


def test_hash_is_sha256_of_text():
    c = Charter.from_text(SEVEN)
    assert c.hash == "0x" + hashlib.sha256(SEVEN.encode("utf-8")).hexdigest()
    assert DEFAULT_CHARTER.hash == "0x" + hashlib.sha256(V1_TEXT.encode()).hexdigest()


def test_charter_hash_follows_active_charter(seven_axis):
    from nodes.common.world_model_substrate.charter_version import (
        charter_hash, charter_payload)
    assert "0x" + charter_hash() == seven_axis.hash
    assert charter_payload()["n_dims"] == 7


def test_default_is_active_by_default():
    assert active_charter() is DEFAULT_CHARTER


# ---------------------------------------------------------------------------
# World, scorer, drift under a custom charter
# ---------------------------------------------------------------------------


def test_custom_charter_world_has_seven_roots(seven_axis):
    from nodes.common.world_model_substrate.adapter import build_charter_world
    world = build_charter_world(embedding_dim=4)
    assert list(world.tendencies.keys()) == list(seven_axis.ids)
    for i, tid in enumerate(seven_axis.ids):
        anchor = world.tendencies[tid].anchor
        assert len(anchor) == 7 + 4
        assert anchor[i] == 1.0 and sum(anchor) == 1.0


def test_score_turn_sized_to_charter(seven_axis):
    from nodes.common.world_model_substrate.score_turn import score_turn
    turn = {"content": "rm -rf / without backup, delete everything, force push"}
    set_active_charter(DEFAULT_CHARTER)
    v1 = score_turn(turn)
    set_active_charter(seven_axis)
    seven = score_turn(turn)
    assert len(v1) == 6 and len(seven) == 7
    assert seven[1] == 0.0                      # unknown axis abstains
    by_id = dict(zip(seven_axis.ids, seven))
    for i, aid in enumerate(DEFAULT_CHARTER.ids):
        assert by_id[aid] == v1[i]               # known axes unchanged


def test_turn_to_observation_sized_to_charter(seven_axis):
    from nodes.common.world_model_substrate.adapter import turn_to_observation
    obs = turn_to_observation({"correctness_impact": 0.5, "kindness_impact": 0.25})
    assert len(obs.coords) == 7
    assert obs.coords[seven_axis.axis_index["correctness"]] == 0.5
    assert obs.coords[seven_axis.axis_index["kindness"]] == 0.25


def test_llm_score_turn_sized_to_charter(seven_axis):
    import asyncio

    from nodes.common.world_model_substrate.llm_score_turn import llm_score_turn

    class _Resp:
        text = json.dumps({"life_precious": 1, "self_preservation": 0,
                           "promotion_of_intelligence": 0, "evolution": -1,
                           "correctness": 1, "simplicity": 0})

    class _Prov:
        name = "fake"

        async def send(self, **kw):
            return _Resp()

    out = asyncio.run(llm_score_turn({"content": "x"}, _Prov(), n_samples=1))
    assert len(out) == 7
    by_id = dict(zip(seven_axis.ids, out))
    assert by_id["kindness"] == 0.0
    assert by_id["life_precious"] == 1.0
    assert by_id["evolution"] == -1.0
    assert by_id["correctness"] == 1.0


DIGEST = "ab" * 32
AUTHOR = "toolsmith"
EMBED_DIM = 8


def _registration(n_dims: int) -> Dict[str, Any]:
    coords = [0.0] * (n_dims + EMBED_DIM)
    coords[n_dims + 3] = 0.5
    return {
        "kind": "sub_claim_sprouted", "seq": 1, "author_agent": AUTHOR,
        "tendency_id": "correctness", "parent_id": "solver_root",
        "node_id": f"tm_{DIGEST[:12]}", "position": "pro",
        "coords": coords, "polarity_axis": coords,
        "content": "tool: does a thing", "author_post": True,
        "artifact_digest": DIGEST, "observation_id": "tm_" + DIGEST[:16],
        "manifest_meta": {"trust_class": "pinned", "author": AUTHOR},
    }


def test_ratings_under_custom_charter(seven_axis):
    """Positions follow the active charter's axes: a ToolBond rating on a
    custom axis lands in its slot, an axis outside the charter is dropped,
    and a carried 6-length (v1) position is replaced by the 7-axis one."""
    from nodes.common.federated_reconcile import compute_tool_mint
    from nodes.common.world_model_substrate.adapter import build_charter_world
    from nodes.common.world_model_substrate.aggregate import apply_events

    world = build_charter_world(embedding_dim=EMBED_DIM)
    apply_events(world, [_registration(7)], equilibrate_after=False)
    regs = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}
    snap = {"supply": 100, "tools": {DIGEST: {
        "axes": {"kindness": {"sum": 60 * 10, "mass": 10},
                 "correctness": {"sum": 80 * 20, "mass": 20},
                 "honesty": {"sum": 50, "mass": 1}},
        "harm": {"sum": 0, "mass": 0}}}}
    out = compute_tool_mint(world, [], registrations=regs, review_snapshot=snap,
                            positions={DIGEST: {"head": [0.1] * 6, "mass": [2.0] * 6}})
    pos = out["positions_next"][DIGEST]
    assert len(pos["head"]) == 7 and len(pos["mass"]) == 7
    k = seven_axis.axis_index["kindness"]
    c = seven_axis.axis_index["correctness"]
    assert pos["head"][k] == pytest.approx(0.6)
    assert pos["head"][c] == pytest.approx(0.8)
    assert pos["mass"][k] == pytest.approx(0.1) and pos["mass"][c] == pytest.approx(0.2)
    for i in range(7):
        if i not in (k, c):
            assert pos["head"][i] == 0.0 and pos["mass"][i] == 0.0


# ---------------------------------------------------------------------------
# Pinned charter replay + the close driver's restart-required signal
# ---------------------------------------------------------------------------

KEY = "jurisdiction.autonet.charter"


def test_pinned_replay_takes_last_value_at_or_before_block():
    from nodes.common.voice_state import charter_text_from_logs
    recs = [
        {"block": 10, "index": 0, "key": KEY, "value": V1_TEXT},
        {"block": 20, "index": 3, "key": "jurisdiction.autonet.epoch_rules", "value": "1"},
        {"block": 20, "index": 1, "key": KEY, "value": SEVEN},
        {"block": 30, "index": 0, "key": KEY, "value": "{\"axes\":\"later\"}"},
    ]
    assert charter_text_from_logs(recs, 9) is None
    assert charter_text_from_logs(recs, 10) == V1_TEXT
    assert charter_text_from_logs(recs, 19) == V1_TEXT
    assert charter_text_from_logs(recs, 20) == SEVEN
    assert charter_text_from_logs(recs, 29) == SEVEN
    # Same block: log index orders the writes.
    same = [{"block": 5, "index": 2, "key": KEY, "value": SEVEN},
            {"block": 5, "index": 1, "key": KEY, "value": V1_TEXT}]
    assert charter_text_from_logs(same, 5) == SEVEN
    # A legacy hash value is not charter text.
    assert charter_text_from_logs(
        [{"block": 1, "index": 0, "key": KEY, "value": "0x" + "ab" * 32}], 1) is None


def test_read_pinned_charter_text_via_logs(monkeypatch):
    from nodes.common import voice_state

    logs = [
        {"blockNumber": 4, "logIndex": 0, "args": {"key": KEY, "value": V1_TEXT}},
        {"blockNumber": 8, "logIndex": 0, "args": {"key": KEY, "value": SEVEN}},
    ]
    seen = {}

    def fake_get_logs(event, w3, address, to_block, start_block=None):
        seen["to_block"] = to_block
        return [lg for lg in logs if lg["blockNumber"] <= to_block]

    monkeypatch.setattr(voice_state, "_get_logs", fake_get_logs)

    class _Events:
        RegistryUpdated = object()

    class _Contract:
        address = "0x" + "11" * 20
        events = _Events()

    class _Eth:
        def contract(self, address, abi):
            assert any(e["name"] == "RegistryUpdated" for e in abi)
            return _Contract()

    class _W3:
        eth = _Eth()

    out = voice_state.read_pinned_charter_text(_W3(), "0x" + "11" * 20, 6)
    assert out == V1_TEXT and seen["to_block"] == 6


class _Gossip:
    def __init__(self):
        self.drained = False

    def drain_epoch_batches(self):
        self.drained = True
        return ["batch"]


def test_driver_refuses_close_and_signals_on_charter_change():
    from nodes.common.federated_close_driver import FederatedCloseDriver

    drv = FederatedCloseDriver(_Gossip())
    calls = []
    drv.on_charter_change = calls.append
    drv.voice_source = lambda: {"snapshot_block": 7, "charter_text": SEVEN}
    assert drv.run({"epoch_id": "e1"}) is None
    assert drv.charter_restart_required == SEVEN
    assert calls == [SEVEN]
    # Stays halted; the callback fires once.
    assert drv.run({"epoch_id": "e2"}) is None
    assert calls == [SEVEN]


def test_driver_runs_when_pinned_charter_matches(monkeypatch):
    from nodes.common import federated_close_driver as fcd

    drv = fcd.FederatedCloseDriver(_Gossip())
    drv.voice_source = lambda: {"snapshot_block": 7, "charter_text": V1_TEXT}
    reached = {}

    def fake_order(batches):
        reached["ordered"] = True
        raise RuntimeError("stop here")

    monkeypatch.setattr(fcd, "canonical_order", fake_order)
    with pytest.raises(RuntimeError, match="stop here"):
        drv.run({"epoch_id": "e1"})
    assert reached and drv.charter_restart_required is None
