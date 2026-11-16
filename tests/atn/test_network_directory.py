"""The published list of networks and jurisdictions (atn/network_directory.py)
and how a choice made from it reaches the daemon: the data source supplies
the Registry, the RPC and the network's label facts; the chain has to agree;
the daemon keeps no table of chains."""
from __future__ import annotations

import json

import pytest

from atn import config as cfg
from atn import network_directory as nd

BASE = "http://data.test"
FACTORY = "0x" + "fa" * 20
DAO_A, REG_A = "0x" + "A1" * 20, "0x" + "a2" * 20
DAO_B, REG_B = "0x" + "b1" * 20, "0x" + "b2" * 20
DAO_PLAIN = "0x" + "c1" * 20


def _files():
    daos_old = {"coll": "daos", "docs": [
        {"id": DAO_PLAIN, "v": 1, "data": {"name": "No Autonet", "registryAddress": REG_B,
                                           "registry": {"other.key": "x"}}},
        {"id": DAO_B, "v": 2, "data": {
            "name": "Beta", "address": DAO_B, "registryAddress": REG_B,
            "registry": {nd.SUBSTRATE_KEY: "0x" + "5b" * 20}}},
    ]}
    daos_new = {"coll": "daos", "docs": [
        {"id": DAO_A, "v": 3, "data": {
            "name": "alpha", "registryAddress": REG_A,
            "registry": {nd.SUBSTRATE_KEY: "0x" + "5a" * 20}}},
        {"id": "0x" + "d1" * 20, "v": 4, "deleted": True},
    ]}
    return {
        f"{BASE}/networks.json": {"format": 1, "networks": [
            {"id": "Test-Net", "data": {
                "chainId": 31337, "rpc": "http://rpc.test", "symbol": "TST",
                "nativeCurrency": "Test Coin", "blockExplorer": "http://scan.test",
                "jurisdictionFactory": FACTORY}},
            {"id": "No-Factory", "data": {"chainId": 5, "rpc": "http://x"}},
            {"id": "Other-Net", "data": {
                "chainId": "84532", "rpc": "http://rpc.other", "nativeCurrency": "ETH",
                "jurisdictionFactory": FACTORY}},
        ]},
        f"{BASE}/Test-Net/ptr.json": {"format": 1, "head": "aaaaaaaa", "seq": 1},
        f"{BASE}/Test-Net/o/aaaaaaaa.json": {"format": 1, "lists": {"daos": {
            "coll": "daos", "count": 3,
            "pages": [{"h": "bbbbbbbb", "n": 2}, {"h": "cccccccc", "n": 1}]}}},
        f"{BASE}/Test-Net/o/bbbbbbbb.json": daos_old,
        f"{BASE}/Test-Net/o/cccccccc.json": daos_new,
        f"{BASE}/Other-Net/ptr.json": {"format": 1, "head": None, "seq": 1},
    }


@pytest.fixture
def source(monkeypatch):
    files = _files()
    fetched: list[str] = []

    def get(url):
        fetched.append(url)
        if url not in files:
            raise nd.DirectoryError(f"can't read {url}: 404")
        return json.loads(json.dumps(files[url]))

    monkeypatch.setattr(nd, "_get", get)
    monkeypatch.setattr(nd, "_OBJECTS", {})
    monkeypatch.delenv("ATN_DATA_URL", raising=False)
    return fetched


def test_data_url_precedence(monkeypatch):
    monkeypatch.delenv("ATN_DATA_URL", raising=False)
    assert nd.data_url() == nd.DEFAULT_DATA_URL
    assert nd.data_url("http://mine/") == "http://mine"
    monkeypatch.setenv("ATN_DATA_URL", "http://env")
    assert nd.data_url("http://mine") == "http://env"


def test_networks_are_the_ones_that_can_hold_jurisdictions(source):
    nets = nd.list_networks(BASE)
    assert [n["name"] for n in nets] == ["Other-Net", "Test-Net"]
    test = nets[1]
    assert test == {"name": "Test-Net", "chain_id": 31337, "rpc_url": "http://rpc.test",
                    "gas_symbol": "TST", "explorer": "http://scan.test"}
    # The symbol falls back to nativeCurrency; the chain id may be a string.
    assert nets[0]["gas_symbol"] == "ETH" and nets[0]["chain_id"] == 84532
    assert nd.get_network("test-net", BASE)["name"] == "Test-Net"
    with pytest.raises(nd.DirectoryError, match="Other-Net, Test-Net"):
        nd.get_network("Nope", BASE)


def test_jurisdictions_page_search_and_skip_plain_daos(source):
    page = nd.list_jurisdictions("Test-Net", BASE)
    assert page["total"] == 2 and page["network"] == "Test-Net"
    assert [j["name"] for j in page["jurisdictions"]] == ["alpha", "Beta"]
    assert page["jurisdictions"][0] == {"name": "alpha", "dao": DAO_A, "dao_registry": REG_A}

    one = nd.list_jurisdictions("Test-Net", BASE, limit=1, offset=1)
    assert [j["name"] for j in one["jurisdictions"]] == ["Beta"] and one["total"] == 2
    assert nd.list_jurisdictions("Test-Net", BASE, query="BET")["total"] == 1
    assert nd.list_jurisdictions("Test-Net", BASE, query=DAO_A[:8].lower())["total"] == 1
    # A network whose scope was never published, or is empty, has none.
    assert nd.list_jurisdictions("Other-Net", BASE)["total"] == 0


def test_hashed_objects_are_fetched_once(source):
    nd.list_jurisdictions("Test-Net", BASE)
    nd.list_jurisdictions("Test-Net", BASE)
    assert source.count(f"{BASE}/Test-Net/o/bbbbbbbb.json") == 1
    assert source.count(f"{BASE}/Test-Net/ptr.json") == 2     # the pointer moves


def test_find_jurisdiction(source):
    found = nd.find_jurisdiction("Test-Net", DAO_A.lower(), BASE)
    assert found["dao_registry"] == REG_A and found["network"]["chain_id"] == 31337
    with pytest.raises(nd.DirectoryError, match="no Autonet jurisdiction"):
        nd.find_jurisdiction("Test-Net", DAO_PLAIN, BASE)
    with pytest.raises(nd.DirectoryError, match="not a DAO address"):
        nd.find_jurisdiction("Test-Net", "alpha", BASE)


def _registry_doc(dao, chain_id=31337):
    from nodes.common.library.charter import DEFAULT_CHARTER
    charter = json.dumps({"axes": [{"id": c["id"], "thesis": c["thesis"]}
                                   for c in DEFAULT_CHARTER.entries()]},
                         separators=(",", ":"))
    return {"jurisdictions": {"autonet": {
        "network": {"rpc_url": "http://rpc.test", "chain_id": chain_id},
        "contracts": {"dao": dao, "timelock": "0x71", "rep_token": "0x2E9",
                      "tool_bond": "0xB0", "substrate": "0x5B", "charter": charter}}}}


def test_resolve_takes_the_registry_and_facts_from_the_source(source, monkeypatch):
    seen = {}

    def fetch(registry, rpc, jid):
        seen.update(registry=registry, rpc=rpc)
        return _registry_doc(DAO_A)

    monkeypatch.setattr(cfg, "_fetch_onchain_registry", fetch)
    monkeypatch.setattr(cfg, "dao_name", lambda dao, rpc: "")
    check = cfg.resolve_jurisdiction("Test-Net", DAO_A, BASE)
    assert check["ok"], check["errors"]
    assert seen == {"registry": REG_A, "rpc": "http://rpc.test"}
    assert check["network"] == {"name": "Test-Net", "gas_symbol": "TST",
                                "explorer": "http://scan.test"}
    assert check["name"] == "alpha" and check["chain_id"] == 31337


def test_resolve_refuses_when_the_chain_disagrees(source, monkeypatch):
    monkeypatch.setattr(cfg, "dao_name", lambda dao, rpc: "")
    # The RPC answers for another chain than the one the source lists.
    monkeypatch.setattr(cfg, "_fetch_onchain_registry",
                        lambda *a: _registry_doc(DAO_A, chain_id=1))
    bad = cfg.resolve_jurisdiction("Test-Net", DAO_A, BASE)
    assert not bad["ok"] and "chain 1" in " ".join(bad["errors"])
    # The Registry the source names belongs to another DAO.
    monkeypatch.setattr(cfg, "_fetch_onchain_registry", lambda *a: _registry_doc(DAO_B))
    bad = cfg.resolve_jurisdiction("Test-Net", DAO_A, BASE)
    assert not bad["ok"] and "belongs to DAO" in " ".join(bad["errors"])
    # Not listed at all: nothing is read on chain.
    monkeypatch.setattr(cfg, "_fetch_onchain_registry",
                        lambda *a: pytest.fail("read the chain for an unlisted DAO"))
    bad = cfg.resolve_jurisdiction("Test-Net", DAO_PLAIN, BASE)
    assert not bad["ok"] and "no Autonet jurisdiction" in bad["errors"][0]


def test_direct_join_carries_only_the_facts_given(monkeypatch):
    monkeypatch.setattr(cfg, "_fetch_onchain_registry", lambda *a: _registry_doc(DAO_A))
    monkeypatch.setattr(cfg, "dao_name", lambda dao, rpc: "")
    bare = cfg.check_jurisdiction(REG_A, "http://rpc.test")
    assert bare["ok"] and bare["network"] == {}
    named = cfg.check_jurisdiction(REG_A, "http://rpc.test", network={
        "name": "My Chain", "gas_symbol": " MYC ", "explorer": None})
    assert named["network"] == {"name": "My Chain", "gas_symbol": "MYC"}


def test_the_facts_are_saved_and_reach_the_config(tmp_path, monkeypatch):
    sel = {"chain_id": 31337, "rpc_url": "http://rpc.test", "dao_registry": REG_A,
           "dao": DAO_A, "name": "alpha",
           "network": {"name": "Test-Net", "gas_symbol": "TST", "explorer": "http://scan.test",
                       "chain_id": 31337}}
    cfg.save_selected_jurisdiction(sel, tmp_path)
    saved = json.loads((tmp_path / "jurisdiction.json").read_text())
    assert saved["network"] == {"name": "Test-Net", "gas_symbol": "TST",
                                "explorer": "http://scan.test"}

    monkeypatch.setattr(cfg, "_SELECTION_DATA_DIR", tmp_path)
    monkeypatch.delenv("ATN_JURISDICTION_REGISTRY", raising=False)
    monkeypatch.delenv("ATN_REGISTRY_URL", raising=False)
    monkeypatch.setattr(cfg, "_fetch_onchain_registry", lambda *a: _registry_doc(DAO_A))
    an = cfg.RPBConfig()
    cfg.resolve_network_registry(an, force=True)
    assert (an.network_name, an.gas_symbol, an.block_explorer) == (
        "Test-Net", "TST", "http://scan.test")


def test_no_facts_means_no_guess(tmp_path, monkeypatch):
    """A jurisdiction named by Registry and RPC alone: the daemon labels gas
    "gas" rather than guess a symbol from the chain id."""
    from atn.autonet_service import AutonetBridge, AutonetState
    cfg.save_selected_jurisdiction(
        {"chain_id": 127823, "rpc_url": "http://rpc.test", "dao_registry": REG_A,
         "dao": DAO_A}, tmp_path)
    assert json.loads((tmp_path / "jurisdiction.json").read_text())["network"] == {}
    assert cfg.RPBConfig().gas_symbol == "" and cfg.RPBConfig().gas_decimals == 18
    assert not hasattr(cfg, "gas_symbol_for_chain")

    svc = AutonetBridge.__new__(AutonetBridge)
    svc.config = cfg.RPBConfig(chain_id=127823)
    svc.state = AutonetState()
    try:
        svc._mirror_jurisdiction()
    except Exception:
        pass        # only the label lines at the top matter here
    assert svc.state.gas_symbol == "gas" and svc.state.network_name == ""
