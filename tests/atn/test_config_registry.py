"""Tests for registry resolution and agents_dir defaulting in atn.config.

  Fix 1 - network registry: packaged with the release (atn/registry.json,
    identical to the repo-root copy), never fetched at boot, resolved when
    the daemon joins the network (first registration or autonet.enabled).
    ATN_REGISTRY_URL is the one network source, fetched at join time only.
    Every test here fails on any unexpected fetch.

  Fix 2 - agents_dir default:
    ./agents in CWD wins (back-compat); otherwise <data_dir>/agents. An
    explicit agents_dir in config.yaml is always honored verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atn import config as cfg


# ---------------------------------------------------------------------------
# Fix 1: network registry
# ---------------------------------------------------------------------------

_SAMPLE = {
    "version": 1,
    "jurisdictions": {
        "autonet": {
            "name": "Autonet",
            "network": {
                "rpc_url": "https://rpc.example.test",
                "chain_id": 424242,
                "gas_symbol": "ZZZ",
                "gas_decimals": 18,
            },
            "contracts": {
                "dao": "0xDA0000000000000000000000000000000000dead",
                "substrate": "0x5UB000000000000000000000000000000000beef",
                "charter_anchor": "0xC4A000000000000000000000000000000000face",
                "service_registry": "0x5E4000000000000000000000000000000000cafe",
            },
        }
    },
}


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """Point the packaged registry at a tmp copy of _SAMPLE, clear the
    ATN_REGISTRY_URL override, and fail on any network fetch."""
    packaged = tmp_path / "pkg" / "registry.json"
    packaged.parent.mkdir(parents=True)
    packaged.write_text(json.dumps(_SAMPLE), encoding="utf-8")
    monkeypatch.setattr(cfg, "_packaged_registry_path", lambda: packaged)
    monkeypatch.delenv("ATN_REGISTRY_URL", raising=False)
    fetched: list[str] = []

    def _no_network(url):
        fetched.append(url)
        raise AssertionError(f"unexpected registry fetch: {url}")

    monkeypatch.setattr(cfg, "_fetch_registry", _no_network)
    return {"packaged": packaged, "fetched": fetched, "tmp": tmp_path}


def _no_dotenv(monkeypatch):
    # Avoid touching the real ~/.atn env dotfile.
    monkeypatch.setattr(cfg, "_load_dotenv", lambda *a, **k: 0)


def test_packaged_registry_matches_repo_copy():
    """atn/registry.json (package data, pinned per release) must equal the
    repo-root registry.json of record. Compared parsed, so line endings of a
    Windows checkout don't matter. Re-sync: copy registry.json to atn/."""
    packaged = Path(cfg.__file__).resolve().parent / "registry.json"
    repo_copy = packaged.parent.parent / "registry.json"
    assert packaged.is_file(), "atn/registry.json missing from the package"
    if not repo_copy.is_file():
        pytest.skip("not a source checkout (no repo-root registry.json)")
    assert (json.loads(packaged.read_text(encoding="utf-8"))
            == json.loads(repo_copy.read_text(encoding="utf-8"))), (
        "atn/registry.json is stale: copy the repo-root registry.json over it")


def test_pyproject_ships_registry_as_package_data():
    pyproject = Path(cfg.__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("not a source checkout")
    import tomllib
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert "registry.json" in data["tool"]["setuptools"]["package-data"]["atn"]


def test_seed_reads_packaged_copy(_isolate_registry):
    seed = cfg._load_registry_seed("autonet")

    assert seed["rpc_url"] == "https://rpc.example.test"
    assert seed["chain_id"] == 424242
    assert seed["dao_address"] == "0xDA0000000000000000000000000000000000dead"
    assert seed["substrate_address"] == "0x5UB000000000000000000000000000000000beef"
    assert seed["charter_anchor_address"] == "0xC4A000000000000000000000000000000000face"
    assert seed["registry_address"] == "0x5E4000000000000000000000000000000000cafe"
    assert _isolate_registry["fetched"] == []


def test_degraded_empty_with_warning(_isolate_registry, caplog):
    """Packaged copy missing -> {} with exactly one warning, no fetch."""
    _isolate_registry["packaged"].unlink()
    with caplog.at_level("WARNING"):
        seed = cfg._load_registry_seed("autonet")

    assert seed == {}
    warnings = [r for r in caplog.records
                if "network registry unavailable" in r.getMessage()]
    assert len(warnings) == 1
    assert _isolate_registry["fetched"] == []


def test_unknown_jurisdiction_returns_empty(_isolate_registry):
    assert cfg._load_registry_seed("nonexistent-guild") == {}


def test_no_network_call_at_boot_without_config(_isolate_registry, tmp_path,
                                                monkeypatch):
    """A fresh install (no config.yaml) boots fully local: no registry fetch,
    no chain addresses, not joined."""
    _no_dotenv(monkeypatch)
    an = cfg.load_config(tmp_path / "missing.yaml").autonet

    assert _isolate_registry["fetched"] == []
    assert an.network_joined is False
    assert an.substrate_address == ""
    assert an.service_registry_address == ""
    assert an.enabled is False  # Phase 12: still starts on registration


def test_no_network_call_at_boot_even_with_override(_isolate_registry,
                                                     tmp_path, monkeypatch):
    """ATN_REGISTRY_URL is read at join time, never at boot."""
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("ATN_REGISTRY_URL", "https://fork.example/registry.json")
    an = cfg.load_config(tmp_path / "missing.yaml").autonet

    assert _isolate_registry["fetched"] == []
    assert an.network_joined is False


def test_registry_resolved_on_join_from_packaged_copy(_isolate_registry,
                                                      tmp_path, monkeypatch):
    """Joining (first registration / autonet start) fills the chain
    addresses from the packaged copy, without any fetch."""
    _no_dotenv(monkeypatch)
    an = cfg.load_config(tmp_path / "missing.yaml").autonet
    assert cfg.resolve_network_registry(an) is True

    assert an.network_joined is True
    assert an.rpc_url == "https://rpc.example.test"
    assert an.chain_id == 424242
    assert an.gas_symbol == "ZZZ"
    assert an.substrate_address == "0x5UB000000000000000000000000000000000beef"
    assert an.service_registry_address == "0x5E4000000000000000000000000000000000cafe"
    assert _isolate_registry["fetched"] == []


def test_registry_resolved_when_enabled(_isolate_registry, tmp_path, monkeypatch):
    """``autonet.enabled: true`` joins at load time (packaged copy, no fetch)."""
    _no_dotenv(monkeypatch)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("autonet:\n  enabled: true\n", encoding="utf-8")
    an = cfg.load_config(cfg_file).autonet

    assert an.enabled is True
    assert an.network_joined is True
    assert an.substrate_address == "0x5UB000000000000000000000000000000000beef"
    assert _isolate_registry["fetched"] == []


def test_explicit_config_wins_over_registry(_isolate_registry, tmp_path,
                                            monkeypatch):
    _no_dotenv(monkeypatch)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "autonet:\n  rpc_url: https://my.rpc\n"
        "  substrate_address: '0xMINE'\n", encoding="utf-8")
    an = cfg.load_config(cfg_file).autonet
    cfg.resolve_network_registry(an)

    assert an.rpc_url == "https://my.rpc"
    assert an.substrate_address == "0xMINE"
    # Unset fields still come from the registry.
    assert an.service_registry_address == "0x5E4000000000000000000000000000000000cafe"


def test_registry_url_override_used_on_join(_isolate_registry, monkeypatch):
    """ATN_REGISTRY_URL, when set, is fetched at join time and wins over the
    packaged copy."""
    fork = json.loads(json.dumps(_SAMPLE))
    fork["jurisdictions"]["autonet"]["contracts"]["substrate"] = "0xF0RK"
    seen: list[str] = []

    def _fake_fetch(url):
        seen.append(url)
        return fork

    monkeypatch.setattr(cfg, "_fetch_registry", _fake_fetch)
    monkeypatch.setenv("ATN_REGISTRY_URL", "https://fork.example/registry.json")
    an = cfg.RPBConfig()
    cfg.resolve_network_registry(an)

    assert seen == ["https://fork.example/registry.json"]
    assert an.substrate_address == "0xF0RK"


def test_registry_url_override_falls_back_to_packaged(_isolate_registry,
                                                      monkeypatch, caplog):
    monkeypatch.setattr(cfg, "_fetch_registry", lambda url: None)
    monkeypatch.setenv("ATN_REGISTRY_URL", "http://127.0.0.1:1/registry.json")
    with caplog.at_level("WARNING"):
        seed = cfg._load_registry_seed("autonet")

    assert seed["substrate_address"] == "0x5UB000000000000000000000000000000000beef"
    assert any("ATN_REGISTRY_URL" in r.getMessage() for r in caplog.records)


def test_bridge_ensure_network_config_joins(_isolate_registry):
    """The registration path's join hook resolves the registry and refreshes
    the bridge state the UI reads."""
    from atn.autonet_service import AutonetBridge

    bridge = AutonetBridge(cfg.RPBConfig())
    assert bridge.network_joined is False
    assert bridge.state.substrate_address == ""

    assert bridge.ensure_network_config() is True
    assert bridge.network_joined is True
    assert bridge.state.substrate_address == "0x5UB000000000000000000000000000000000beef"
    assert bridge.state.chain_id == 424242
    assert _isolate_registry["fetched"] == []


# ---------------------------------------------------------------------------
# Fix 2: agents_dir defaulting
# ---------------------------------------------------------------------------

def test_agents_dir_defaults_to_data_dir(tmp_path, monkeypatch):
    """No ./agents in CWD -> default is <data_dir>/agents (no stray mkdir)."""
    work = tmp_path / "neutral"
    work.mkdir()
    monkeypatch.chdir(work)

    resolved = cfg._default_agents_dir(tmp_path / "datadir")
    assert resolved == tmp_path / "datadir" / "agents"
    # Resolving must not create a stray ./agents in the CWD.
    assert not (work / "agents").exists()


def test_agents_dir_uses_cwd_agents_when_present(tmp_path, monkeypatch):
    """A ./agents dir in CWD wins for back-compat."""
    work = tmp_path / "repo"
    work.mkdir()
    (work / "agents").mkdir()
    monkeypatch.chdir(work)

    resolved = cfg._default_agents_dir(tmp_path / "datadir")
    assert resolved == Path("agents")


def test_load_config_no_file_uses_home_rooted_agents(tmp_path, monkeypatch):
    """load_config with no config file defaults agents_dir off the data dir,
    not the launch CWD."""
    work = tmp_path / "launch"
    work.mkdir()
    monkeypatch.chdir(work)
    # Avoid touching the real ~/.atn env dotfile.
    monkeypatch.setattr(cfg, "_load_dotenv", lambda: None)

    conf = cfg.load_config(path=tmp_path / "does-not-exist.yaml")

    # Default data_dir is ~/.atn; agents_dir must live under it.
    assert conf.agents_dir == cfg._DEFAULT_DIR / "agents"
    assert not (work / "agents").exists()


def test_load_config_explicit_agents_dir_honored(tmp_path, monkeypatch):
    """An explicit agents_dir in config.yaml is honored verbatim (relative
    to the config file location) and unaffected by the default logic."""
    monkeypatch.setattr(cfg, "_load_dotenv", lambda: None)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text("agents_dir: my_agents\n", encoding="utf-8")

    conf = cfg.load_config(path=cfg_file)

    assert conf.agents_dir == (cfg_dir / "my_agents").resolve()


def test_load_config_agents_dir_tracks_custom_data_dir(tmp_path, monkeypatch):
    """With a custom data_dir and no agents_dir, the default tracks data_dir."""
    monkeypatch.setattr(cfg, "_load_dotenv", lambda: None)
    work = tmp_path / "launch"
    work.mkdir()
    monkeypatch.chdir(work)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    data_dir = tmp_path / "mydata"
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text(f"data_dir: {data_dir.as_posix()}\n", encoding="utf-8")

    conf = cfg.load_config(path=cfg_file)

    assert conf.data_dir == data_dir
    assert conf.agents_dir == data_dir / "agents"
