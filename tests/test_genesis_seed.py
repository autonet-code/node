"""Merging by genesis (docs/multi_jurisdiction.md decision 9).

A new DAO's FIRST anchor adopts a posted snapshot: payload ``"seed": true``
whose carry bundle is the merge of the source jurisdictions' latest anchored
bundles. Covers the builder's merge and hashes, the fetch round trip, and the
close driver installing the seed at anchor 0 (no prev_root match, over
existing carry files, exactly once) while ignoring a seed flag later on.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from nodes.common.authoritative_encoding import (
    CARRY_KEYS,
    cid_for_blob,
    decode_carry_bundle,
    encode_authoritative_payload,
    encode_carry_bundle,
)
from nodes.common.governed_anchor import anchor_description
from nodes.common.state_sync import fetch_carry_over_from_chain
from tests.test_carry_over_commitment import _driver, _run

_SPEC = importlib.util.spec_from_file_location(
    "build_genesis_seed",
    Path(__file__).resolve().parent.parent / "scripts" / "build_genesis_seed.py")
seedmod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = seedmod
_SPEC.loader.exec_module(seedmod)

ZERO = "00" * 32


def _carry(regs=None, positions=None, manifests=None, busts=None) -> Dict[str, Any]:
    return {
        "tool_registrations": regs or {},
        "tool_rounds": {"manifests": manifests or {}, "busts": busts or {}},
        "tool_positions": positions or {},
    }


def _src(label, carry):
    return seedmod.SourceCarry(label=label, kind="bundle", carry=carry, verified=False)


# ---------------------------------------------------------------------------
# Fake Substrate + blob source
# ---------------------------------------------------------------------------


class _Call:
    def __init__(self, v):
        self._v = v

    def call(self):
        return self._v


class FakeSubstrate:
    def __init__(self):
        self.anchors: List[tuple] = []
        outer = self

        class _F:
            def anchorCount(self):
                return _Call(len(outer.anchors))

            def getAnchor(self, i):
                return _Call(outer.anchors[i])

        self.functions = _F()

    def anchor(self, epoch_id, epoch_root_hex, payload_hash_hex, mint_cid=""):
        self.anchors.append((
            epoch_id, bytes.fromhex(epoch_root_hex), bytes(32), bytes(32),
            mint_cid, bytes.fromhex(payload_hash_hex), "0x" + "00" * 20,
            len(self.anchors) + 1, 0, bytes(32)))


def _post_seed(contract, blobs, carry, chain_id=31337):
    seed = seedmod.build_seed(carry, chain_id=chain_id)
    blobs[seed["carry_cid"]] = seed["bundle"]
    blobs[seed["payload_cid"]] = seed["payload"]
    a = seed["anchor_args"]
    contract.anchor(a["epochId"], a["epochRoot"][2:], a["payloadHash"][2:])
    return seed


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def test_merge_union_first_wins_and_reports_conflicts():
    a = _carry(regs={"d1": {"author": "x"}, "d2": {"author": "a"}},
               positions={"d1": {"head": [0.1], "mass": [1.0]}},
               manifests={"d1": {"greenlit": True}}, busts={"v": 1})
    b = _carry(regs={"d2": {"author": "b"}, "d3": {"author": "c"},
                     "d1": {"author": "x"}},
               positions={"d3": {"head": [0.2], "mass": [2.0]}},
               manifests={"d1": {"greenlit": False}}, busts={"v": 2, "w": 1})
    m = seedmod.merge_carry([_src("A", a), _src("B", b)])
    assert list(m.carry["tool_registrations"]) == ["d1", "d2", "d3"]
    assert m.carry["tool_registrations"]["d2"] == {"author": "a"}
    assert set(m.carry["tool_positions"]) == {"d1", "d3"}
    assert m.carry["tool_rounds"]["manifests"]["d1"] == {"greenlit": True}
    assert m.carry["tool_rounds"]["busts"] == {"v": 1, "w": 1}
    got = {(c["map"], c["key"], c["kept"], c["dropped"]) for c in m.conflicts}
    # identical d1 registration is a duplicate, not a conflict
    assert got == {("tool_registrations", "d2", "A", "B"),
                   ("tool_rounds.manifests", "d1", "A", "B"),
                   ("tool_rounds.busts", "v", "A", "B")}
    # swapping the order flips the winner
    m2 = seedmod.merge_carry([_src("B", b), _src("A", a)])
    assert m2.carry["tool_registrations"]["d2"] == {"author": "b"}


def test_merge_and_seed_bytes_are_deterministic():
    a = _carry(regs={"z": {"n": 1}, "a": {"n": 2}})
    b = _carry(regs={"m": {"n": 3}})
    s1 = seedmod.build_seed(seedmod.merge_carry([_src("A", a), _src("B", b)]).carry,
                            chain_id=7)
    # Same values, reversed insertion order: identical bytes.
    a_rev = _carry(regs=dict(reversed(list(a["tool_registrations"].items()))))
    s2 = seedmod.build_seed(seedmod.merge_carry([_src("A", a_rev), _src("B", b)]).carry,
                            chain_id=7)
    assert s1["bundle"] == s2["bundle"] and s1["payload"] == s2["payload"]
    assert decode_carry_bundle(s1["bundle"])["epoch_id"] == "seed"
    # chain id is bound into the root
    s3 = seedmod.build_seed(decode_carry_bundle(s1["bundle"])["carry"], chain_id=8)
    assert s3["carry_cid"] == s1["carry_cid"]
    assert s3["anchor_args"]["epochRoot"] != s1["anchor_args"]["epochRoot"]


def test_seed_payload_and_anchor_args():
    seed = seedmod.build_seed(_carry(regs={"d": {"a": "b"}}), chain_id=5)
    payload = json.loads(seed["payload"])
    assert payload["seed"] is True and list(payload)[-1] == "seed"
    assert payload["carry_cid"] == seed["carry_cid"]
    assert payload["epoch_id"] == "seed" and payload["agent_mint"] == {}
    assert payload["prev_epoch_root"] == ZERO
    a = seed["anchor_args"]
    assert a["payloadHash"] == "0x" + cid_for_blob(seed["payload"])
    assert a["prevEpochRoot"] == a["prevAnchorHash"] == a["agentMintRoot"] == "0x" + ZERO
    assert a["agentMintCid"] == ""
    assert a["epochRoot"][2:] == payload["epoch_root"]
    assert seed["description"] == anchor_description("seed", bytes.fromhex(a["epochRoot"][2:]))
    assert seed["description"] == f"Anchor epoch seed {a['epochRoot']}"
    # Ordinary payloads keep their bytes: no seed key unless set.
    plain = encode_authoritative_payload(
        {"carry_cid": "cc" * 32}, epoch_id="e1",
        epoch_root_hex="11" * 32, prev_epoch_root_hex=ZERO)
    assert "seed" not in json.loads(plain)


def test_builder_cli_writes_blobs_and_json(tmp_path):
    bundle = tmp_path / "a.json"
    bundle.write_bytes(encode_carry_bundle(_carry(regs={"d": {"a": "1"}}), epoch_id="e9"))
    out, store = tmp_path / "out", tmp_path / "store"
    assert seedmod.main(["--chain-id", "3", "--source", f"bundle:{bundle}",
                         "--out", str(out), "--blob-store", str(store)]) == 0
    rep = json.loads((out / "seed_anchor.json").read_text(encoding="utf-8"))
    for cid in (rep["carry_cid"], rep["payload_cid"]):
        assert cid_for_blob((out / cid).read_bytes()) == cid
        assert (store / cid).read_bytes() == (out / cid).read_bytes()
    assert rep["sources"][0]["verified_against_chain"] is False
    assert rep["merged_counts"]["tool_registrations"] == 1


# ---------------------------------------------------------------------------
# Fetch round trip
# ---------------------------------------------------------------------------


def test_seed_round_trips_through_fetch():
    src_chain, src_blobs = FakeSubstrate(), {}
    src_carry = _carry(regs={"d1": {"author": "x"}})
    src_bundle = encode_carry_bundle(src_carry, epoch_id="e4")
    src_payload = encode_authoritative_payload(
        {"carry_cid": cid_for_blob(src_bundle)}, epoch_id="e4",
        epoch_root_hex="44" * 32, prev_epoch_root_hex=ZERO)
    src_blobs[cid_for_blob(src_bundle)] = src_bundle
    src_blobs[cid_for_blob(src_payload)] = src_payload
    src_chain.anchor("e4", "44" * 32, cid_for_blob(src_payload))

    loaded = seedmod.load_chain_source(src_chain, src_blobs.get, "A")
    assert loaded.verified and loaded.carry == src_carry
    fetched_src = fetch_carry_over_from_chain(src_chain, src_blobs.get)
    assert fetched_src.seed is False and fetched_src.anchor_index == 0

    new_chain, new_blobs = FakeSubstrate(), {}
    seed = _post_seed(new_chain, new_blobs, seedmod.merge_carry([loaded]).carry)
    f = fetch_carry_over_from_chain(new_chain, new_blobs.get)
    assert f.status == "ok" and f.seed is True and f.anchor_index == 0
    assert f.epoch_id == "seed" and f.carry_cid == seed["carry_cid"]
    assert f.carry == src_carry
    # Tampered bundle is rejected.
    new_blobs[seed["carry_cid"]] = seed["bundle"].replace(b"x", b"y")
    with pytest.raises(ValueError):
        fetch_carry_over_from_chain(new_chain, new_blobs.get)


# ---------------------------------------------------------------------------
# Close driver
# ---------------------------------------------------------------------------


def _source_carry(tmp_path) -> Dict[str, Any]:
    """A source jurisdiction's carry: a close of epoch 1 (registers a tool)."""
    src = _driver(tmp_path / "source")
    assert _run(src, 1) is not None
    return {k: v for k, v in src.carry_over_maps().items()}


def _voice(contract):
    def voice():
        n = len(contract.anchors)
        return {"owner_map": {}, "voice_weights": {},
                "snapshot_block": None if n == 0 else n,
                "snapshot_anchor_count": n}
    return voice


def _joiner(state_dir, contract, blobs):
    d = _driver(state_dir, voice_source=_voice(contract), enforce=True)
    d.carry_source = lambda: fetch_carry_over_from_chain(contract, blobs.get)
    return d


def _control(tmp_path, carry):
    c = _driver(tmp_path / "control")
    c.install_carry_over(carry)
    return c


def test_driver_installs_seed_over_existing_files_without_prev_root(tmp_path):
    carry = _source_carry(tmp_path)
    contract, blobs = FakeSubstrate(), {}
    d = _joiner(tmp_path / "d", contract, blobs)
    # Pre-seed genesis close: carry files exist, never anchored.
    assert _run(d, 3) is not None
    assert all((tmp_path / "d" / f"{k}.json").exists() for k in CARRY_KEYS)
    assert d.carry_over_maps()["tool_registrations"] == {}
    seed = _post_seed(contract, blobs, carry)

    d._last_observed_root = None      # no prior epoch observed at all
    fed = _run(d, 4)
    assert fed is not None
    assert d.installed_seed_cid() == seed["carry_cid"]
    ctrl = _control(tmp_path, carry)
    assert fed.carry_cid == _run(ctrl, 4).carry_cid
    # Installed exactly once: the next close builds on epoch 4's output.
    fed5 = _run(d, 5)
    assert fed5 is not None and fed5.carry_cid == _run(ctrl, 5).carry_cid


def test_fresh_joiner_installs_seed_and_restart_keeps_marker(tmp_path):
    carry = _source_carry(tmp_path)
    contract, blobs = FakeSubstrate(), {}
    seed = _post_seed(contract, blobs, carry)
    d = _joiner(tmp_path / "d", contract, blobs)
    fed = _run(d, 4)                   # first ever epoch, no observation
    ctrl = _control(tmp_path, carry)
    assert fed is not None and fed.carry_cid == _run(ctrl, 4).carry_cid
    # Restart: a new driver on the same state dir must not reinstall.
    d2 = _joiner(tmp_path / "d", contract, blobs)
    assert d2.installed_seed_cid() == seed["carry_cid"]
    assert _run(d2, 5).carry_cid == _run(ctrl, 5).carry_cid


def test_driver_blocks_while_seed_bundle_unretrievable(tmp_path):
    carry = _source_carry(tmp_path)
    contract, blobs = FakeSubstrate(), {}
    d = _joiner(tmp_path / "d", contract, blobs)
    assert _run(d, 3) is not None     # pre-seed files
    seed = _post_seed(contract, blobs, carry)
    held = blobs.pop(seed["carry_cid"])
    assert _run(d, 4) is None          # refuses to close on pre-seed state
    blobs[seed["carry_cid"]] = held
    assert _run(d, 5) is not None and d.installed_seed_cid() == seed["carry_cid"]


def test_seed_flag_past_anchor_zero_is_ignored(tmp_path, caplog):
    carry = _source_carry(tmp_path)
    contract, blobs = FakeSubstrate(), {}
    # Anchor 0: an ordinary close; anchor 1: a payload claiming seed.
    plain_bundle = encode_carry_bundle(_carry(), epoch_id="e1")
    plain_payload = encode_authoritative_payload(
        {"carry_cid": cid_for_blob(plain_bundle)}, epoch_id="e1",
        epoch_root_hex="55" * 32, prev_epoch_root_hex=ZERO)
    blobs[cid_for_blob(plain_bundle)] = plain_bundle
    blobs[cid_for_blob(plain_payload)] = plain_payload
    contract.anchor("e1", "55" * 32, cid_for_blob(plain_payload))
    _post_seed(contract, blobs, carry)

    with caplog.at_level(logging.WARNING):
        f = fetch_carry_over_from_chain(contract, blobs.get)
    assert f.status == "ok" and f.seed is False and f.anchor_index == 1
    assert "only anchor 0 may seed" in caplog.text

    # Existing files: untouched. Missing files: needs the prev_root match.
    d = _joiner(tmp_path / "d", contract, blobs)
    assert _run(d, 3) is None         # first observed epoch, no install
    assert _run(d, 4) is None         # anchor root != observed root
    assert not any((tmp_path / "d" / f"{k}.json").exists() for k in CARRY_KEYS)
    assert d.installed_seed_cid() is None
