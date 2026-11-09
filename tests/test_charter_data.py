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
from typing import Dict, List

import pytest

from nodes.common.library.charter import (
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
    assert [(c["id"], c["thesis"]) for c in DEFAULT_CHARTER.entries()] == \
        list(DEFAULT_CHARTER.axes)
    assert DEFAULT_CHARTER.n_dims == 6


def test_hash_is_sha256_of_text():
    c = Charter.from_text(SEVEN)
    assert c.hash == "0x" + hashlib.sha256(SEVEN.encode("utf-8")).hexdigest()
    assert DEFAULT_CHARTER.hash == "0x" + hashlib.sha256(V1_TEXT.encode()).hexdigest()


def test_charter_hash_follows_active_charter(seven_axis):
    from nodes.common.library.charter_version import (
        charter_hash, charter_payload)
    assert "0x" + charter_hash() == seven_axis.hash
    assert charter_payload()["n_dims"] == 7


def test_default_is_active_by_default():
    assert active_charter() is DEFAULT_CHARTER


DIGEST = "ab" * 32
AUTHOR = "toolsmith"


def test_ratings_under_custom_charter(seven_axis):
    """Ratings follow the active charter's axes: a ToolBond rating on a
    custom axis is kept, an axis outside the charter is dropped, and the
    value is the mean over the charter's axes together."""
    from nodes.common.federated_reconcile import compute_tool_mint

    regs = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}
    snap = {"supply": 100, "tools": {DIGEST: {
        "axes": {"kindness": {"sum": 60 * 10, "mass": 10},
                 "honesty": {"sum": 50, "mass": 1}},
        "fields": {"correctness": {"sum": 80 * 20, "mass": 20}}}}}
    out = compute_tool_mint([], registrations=regs, review_snapshot=snap)
    r = out["ratings_next"][DIGEST]
    assert r["axes"] == {"kindness": pytest.approx(0.6)}
    assert r["value"] == pytest.approx(0.6)
    assert r["correctness"] == pytest.approx(0.8)
    assert r["mass"] == pytest.approx(0.2)


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
