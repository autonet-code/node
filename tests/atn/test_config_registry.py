"""Tests for registry-seed resolution and agents_dir defaulting in atn.config.

Covers the two pip-parity fixes:

  Fix 1 — registry seed resolution order:
    repo-root registry.json (dev override) > GitHub fetch/cache > stale
    cache > {} + one warning. No real network is used: the URL is pointed
    at a dead localhost port, and the repo-root path is monkeypatched so the
    real checked-in registry.json doesn't shadow the cache/degraded branches.

  Fix 2 — agents_dir default:
    ./agents in CWD wins (back-compat); otherwise <data_dir>/agents. An
    explicit agents_dir in config.yaml is always honored verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atn import config as cfg


# ---------------------------------------------------------------------------
# Fix 1: registry seed resolution
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
    """Point the cache at a tmp dir and the fetch at a dead port by default.

    Individual tests override the repo path / URL as needed.
    """
    cache = tmp_path / "cache" / "registry.json"
    monkeypatch.setattr(cfg, "_REGISTRY_CACHE_PATH", cache)
    # Dead localhost port — connection refused is immediate, no real network.
    monkeypatch.setenv("ATN_REGISTRY_URL", "http://127.0.0.1:1/registry.json")
    # By default hide the real checked-in repo registry.json so the
    # cache/degraded branches are actually exercised. Tests that want the
    # repo-override branch re-point this at a real file.
    monkeypatch.setattr(cfg, "_repo_registry_path",
                        lambda: tmp_path / "no-such-repo-registry.json")
    return {"cache": cache, "tmp": tmp_path}


def test_repo_root_file_wins(_isolate_registry, tmp_path, monkeypatch):
    """A repo-root registry.json takes precedence over cache/network."""
    repo_file = tmp_path / "repo" / "registry.json"
    repo_file.parent.mkdir(parents=True)
    repo_file.write_text(json.dumps(_SAMPLE), encoding="utf-8")
    monkeypatch.setattr(cfg, "_repo_registry_path", lambda: repo_file)

    seed = cfg._load_registry_seed("autonet")

    assert seed["rpc_url"] == "https://rpc.example.test"
    assert seed["chain_id"] == 424242
    assert seed["dao_address"] == "0xDA0000000000000000000000000000000000dead"
    assert seed["substrate_address"] == "0x5UB000000000000000000000000000000000beef"
    assert seed["charter_anchor_address"] == "0xC4A000000000000000000000000000000000face"
    assert seed["registry_address"] == "0x5E4000000000000000000000000000000000cafe"
    # Cache must NOT have been written — repo file short-circuits the fetch.
    assert not _isolate_registry["cache"].exists()


def test_cache_used_when_url_unreachable(_isolate_registry):
    """With no repo file and a dead URL, a pre-existing cache is used."""
    cache = _isolate_registry["cache"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(_SAMPLE), encoding="utf-8")

    seed = cfg._load_registry_seed("autonet")

    assert seed["rpc_url"] == "https://rpc.example.test"
    assert seed["substrate_address"] == "0x5UB000000000000000000000000000000000beef"


def test_degraded_empty_with_warning(_isolate_registry, caplog):
    """No repo file, dead URL, no cache -> {} with exactly one warning."""
    with caplog.at_level("WARNING"):
        seed = cfg._load_registry_seed("autonet")

    assert seed == {}
    warnings = [r for r in caplog.records
                if "network registry unavailable" in r.getMessage()]
    assert len(warnings) == 1


def test_unknown_jurisdiction_returns_empty(_isolate_registry):
    """A present registry that lacks the jurisdiction yields {} (from cache)."""
    cache = _isolate_registry["cache"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(_SAMPLE), encoding="utf-8")

    assert cfg._load_registry_seed("nonexistent-guild") == {}


def test_registry_url_override_respected(monkeypatch):
    """ATN_REGISTRY_URL overrides the default fetch URL."""
    monkeypatch.setenv("ATN_REGISTRY_URL", "https://fork.example/registry.json")
    assert cfg._registry_url() == "https://fork.example/registry.json"
    monkeypatch.delenv("ATN_REGISTRY_URL", raising=False)
    assert cfg._registry_url() == cfg._DEFAULT_REGISTRY_URL


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
