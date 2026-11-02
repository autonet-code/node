"""Switching jurisdictions (docs/multi_jurisdiction.md decisions 1 and 6):
the saved selection, per-jurisdiction registration and publish records,
adopted tools parked outside their jurisdiction, and the join check."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atn import config as cfg
from atn.events import EventBus
from atn.models import AgentIdentity
from atn.tool_store import ToolRecord, ToolStore

# The release ships no jurisdiction now; these test the packaged path on a sample.
pytestmark = pytest.mark.usefixtures("sample_packaged_registry")

A = {"chain_id": 31337, "rpc_url": "http://a", "dao_registry": "0xRA", "dao": "0xDAOA"}
B = {"chain_id": 31337, "rpc_url": "http://b", "dao_registry": "0xRB", "dao": "0xDAOB"}
KEY_A = "31337:0xdaoa"
KEY_B = "31337:0xdaob"


def _runtime(data_dir: Path):
    from atn.config import ATNConfig
    from atn.runtime import Runtime
    agents_dir = data_dir.parent / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def test_selection_round_trip_and_boot_key(tmp_path):
    assert cfg.load_selected_jurisdiction(tmp_path) is None
    cfg.save_selected_jurisdiction({**A, "name": "A"}, tmp_path)
    sel = cfg.load_selected_jurisdiction(tmp_path)
    assert sel["dao_registry"] == "0xRA" and sel["name"] == "A"
    assert cfg.boot_jurisdiction_key(tmp_path) == KEY_A


def test_boot_key_without_selection_is_empty(tmp_path, monkeypatch):
    """Issue #50: no selection, no jurisdiction (the packaged one is a
    choice, not a default)."""
    monkeypatch.delenv("ATN_JURISDICTION_REGISTRY", raising=False)
    assert cfg.boot_jurisdiction_key(tmp_path) == ""


def test_boot_key_of_the_packaged_selection(tmp_path):
    cfg.save_selected_jurisdiction({"packaged": True}, tmp_path)
    packaged = json.loads(cfg._packaged_registry_path().read_text())["jurisdictions"]["autonet"]
    assert cfg.boot_jurisdiction_key(tmp_path) == (
        f"{packaged['network']['chain_id']}:{packaged['contracts']['dao'].lower()}")


def test_incomplete_selection_ignored(tmp_path):
    (tmp_path / cfg.SELECTION_FILE).write_text(json.dumps({"dao": "0x1"}))
    assert cfg.load_selected_jurisdiction(tmp_path) is None


def test_selected_but_unreadable_registry_joins_nothing(tmp_path, monkeypatch):
    cfg.save_selected_jurisdiction(A, tmp_path)
    cfg.set_selection_data_dir(tmp_path)
    monkeypatch.delenv("ATN_REGISTRY_URL", raising=False)
    monkeypatch.delenv("ATN_JURISDICTION_REGISTRY", raising=False)
    monkeypatch.setattr(cfg, "_fetch_onchain_registry", lambda *a: None)
    try:
        assert cfg._load_registry_seed() == {}
    finally:
        cfg.set_selection_data_dir(None)


def test_registration_is_per_jurisdiction(tmp_path):
    data_dir = tmp_path / "data"
    cfg.save_selected_jurisdiction(A, data_dir)
    rt = _runtime(data_dir)
    reg = rt.registry
    assert reg.jurisdiction == KEY_A
    ident = AgentIdentity(address=ADDR, registered_on_chain=True,
                          registration_tx="0xtxA")
    vault = _Vault()
    reg._keystore = lambda: vault
    path = reg._identity_path("ag")
    path.parent.mkdir(parents=True, exist_ok=True)
    reg._save_identity("ag", ident, KEY)
    data = json.loads(path.read_text())
    assert data["registrations"] == {KEY_A: "0xtxA"}

    # Switch to B: same identity, not registered there.
    reg.jurisdiction = KEY_B
    loaded = reg._load_identity("ag")
    assert loaded is not None
    ident_b = loaded[0]
    assert ident_b.registered_on_chain is False
    assert ident_b.registrations == {KEY_A: "0xtxA"}

    # Register in B too; both kept. Back to A: registered again.
    ident_b.registered_on_chain, ident_b.registration_tx = True, "0xtxB"
    reg._save_identity("ag", ident_b, KEY)
    reg.jurisdiction = KEY_A
    ident_a = reg._load_identity("ag")[0]
    assert ident_a.registered_on_chain is True
    assert ident_a.registration_tx == "0xtxA"
    assert set(ident_a.registrations) == {KEY_A, KEY_B}


def test_legacy_identity_file_maps_to_current(tmp_path, monkeypatch):
    """A pre-2026-10-17 identity file registered in the packaged
    jurisdiction: the data dir keeps that jurisdiction (issue #50) and the
    flag maps to it."""
    monkeypatch.delenv("ATN_JURISDICTION_REGISTRY", raising=False)
    monkeypatch.delenv("ATN_REGISTRY_URL", raising=False)
    data_dir = tmp_path / "data"
    path = tmp_path / "agents" / "old" / "identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"public_key": "", "address": ADDR,
                                "registered_on_chain": True,
                                "registration_tx": "0xold"}))
    rt = _runtime(data_dir)
    reg = rt.registry
    assert cfg.load_selected_jurisdiction(data_dir)["packaged"]
    vault = _Vault()
    vault.put_secret(reg._key_secret_name("old"), KEY)
    reg._keystore = lambda: vault
    ident = reg._load_identity("old")[0]
    assert ident.registered_on_chain is True
    assert ident.registrations == {reg.jurisdiction: "0xold"}


KEY = "0x" + "11" * 32
try:
    from eth_account import Account as _Account
    ADDR = _Account.from_key(KEY).address
except Exception:  # pragma: no cover
    ADDR = "0x" + "11" * 20


class _Vault:
    def __init__(self):
        self._s = {}

    def put_secret(self, name, value):
        self._s[name] = value

    def get_secret(self, name):
        return self._s[name]


def _row(digest, **kw):
    return {"digest": digest, "grants": [], "enabled": True, "local_author": "x",
            "registered_ts": 1, **kw}


def test_published_and_adopted_records_follow_the_jurisdiction(tmp_path):
    class _Rt:
        jurisdiction = KEY_A
    store = ToolStore(_Rt(), tmp_path / "tools")
    manifests = {"d1": {"name": "authored"}, "d2": {"name": "adopted_a"},
                 "d3": {"name": "adopted_b"}, "d4": {"name": "legacy"}}

    class _Blobs:
        def get_json(self, d):
            return manifests.get(d)
    store._blobs = _Blobs()
    store._blob_store = lambda: store._blobs
    (tmp_path / "tools").mkdir()
    rows = [
        _row("d1", published=True, published_in=[KEY_A, KEY_B]),
        _row("d2", origin="adopted", jurisdiction=KEY_A),
        _row("d3", origin="adopted", jurisdiction=KEY_B),
        _row("d4", published=True),                       # pre-2026-10-17
    ]
    (tmp_path / "tools" / "registry.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    store._load()
    assert set(store._records) == {"d1", "d2", "d4"}     # d3 parked
    assert store._records["d1"].published is True
    assert store._records["d4"].published_in == {KEY_A}

    # Unpublishing in A keeps the B publication.
    store._records["d1"].origin = "authored"
    store._records["d1"].manifest["author"] = "0x" + "33" * 20
    store.set_published("d1", False)
    assert store._records["d1"].published_in == {KEY_B}
    # The parked record survives a persist.
    written = (tmp_path / "tools" / "registry.jsonl").read_text()
    assert '"d3"' in written

    # In B: d3 is back, d2 is parked, d1 is published.
    _Rt.jurisdiction = KEY_B
    store_b = ToolStore(_Rt(), tmp_path / "tools")
    store_b._blobs = _Blobs()
    store_b._blob_store = lambda: store_b._blobs
    store_b._load()
    assert set(store_b._records) == {"d1", "d3", "d4"}
    assert store_b._records["d1"].published is True
    assert store_b._records["d4"].published is False


def test_check_jurisdiction(monkeypatch):
    from nodes.common.world_model_substrate.adapter import CHARTER
    v1 = json.dumps({"axes": [{"id": c["id"], "thesis": c["thesis"]} for c in CHARTER]}, separators=(",", ":"))
    contracts = {"dao": "0xDAOB", "timelock": "0x71", "rep_token": "0x2E9",
                 "tool_bond": "0xB0", "substrate": "0x5B", "charter": v1}
    doc = {"jurisdictions": {"autonet": {
        "network": {"rpc_url": "http://b", "chain_id": 31337}, "contracts": contracts}}}
    monkeypatch.setattr(cfg, "_fetch_onchain_registry", lambda *a: doc)
    ok = cfg.check_jurisdiction("0xRB", "http://b")
    assert ok["ok"] and ok["key"] == KEY_B and ok["dao"] == "0xDAOB"

    contracts.pop("tool_bond")
    bad = cfg.check_jurisdiction("0xRB", "http://b")
    assert not bad["ok"] and "tool_bond" in bad["errors"][0]

    monkeypatch.setattr(cfg, "_fetch_onchain_registry", lambda *a: None)
    assert not cfg.check_jurisdiction("0xRB", "http://b")["ok"]
