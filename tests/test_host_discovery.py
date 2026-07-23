"""Targeted tests for the ported host-discovery subsystem.

No network, no reliance on any real CLI. Covers:
  * EnvCredentialProbe tier-1 / pattern / exclusion behavior, and the hard
    invariant that VALUES never appear in to_dict() output.
  * HostDiscovery cache round-trip (write then reload preserves env_keys /
    importable).
  * validate_secret_name (the secrets_import name-validation helper).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from atn.runtime.host_discovery import HostDiscovery, validate_secret_name
from atn.runtime.host_discovery.probes import EnvCredentialProbe


def _run(coro):
    return asyncio.run(coro)


# ── EnvCredentialProbe ────────────────────────────────────────────────

def test_env_probe_tier1_wellknown(monkeypatch):
    monkeypatch.setattr("os.environ", {
        "ANTHROPIC_API_KEY": "sk-ant-SUPERSECRETVALUE123",
        "GITHUB_TOKEN": "ghp_TOTALLYSECRET456",
        "HOME": "/home/user",
    }, raising=False)
    r = _run(EnvCredentialProbe().probe())
    assert r.available is True
    assert set(r.env_keys) == {"ANTHROPIC_API_KEY", "GITHUB_TOKEN"}
    assert "2 credential-shaped" in r.detail


def test_env_probe_pattern_sweep(monkeypatch):
    monkeypatch.setattr("os.environ", {
        "MYSERVICE_API_KEY": "abcdefghij",       # matches, len >= 8
        "SOME_TOKEN": "longenoughvalue",          # matches
        "DB_PASSWORD": "hunter2hunter2",          # matches
        "CUSTOM_SECRET": "yes",                   # matches name but value < 8
        "RANDOM_VAR": "notacredential",           # name doesn't match
    }, raising=False)
    r = _run(EnvCredentialProbe().probe())
    assert set(r.env_keys) == {"MYSERVICE_API_KEY", "SOME_TOKEN", "DB_PASSWORD"}
    # short value excluded
    assert "CUSTOM_SECRET" not in r.env_keys


def test_env_probe_exclusions(monkeypatch):
    monkeypatch.setattr("os.environ", {
        "ATN_SOME_TOKEN": "shouldbeexcluded01",    # ATN_ prefix excluded
        "KEYSTORE_SECRET": "shouldbeexcluded02",    # KEYSTORE_ prefix excluded
        "PATH": "/usr/bin:/bin:/opt/x",             # PATH-like name excluded
        "MY_PATH": "/a/b:/c/d",                     # endswith PATH excluded
        "REAL_API_KEY": "keepthisone12345",         # kept
    }, raising=False)
    r = _run(EnvCredentialProbe().probe())
    assert r.env_keys == ["REAL_API_KEY"]


def test_env_probe_tier1_not_double_counted(monkeypatch):
    # A well-known name also matches the generic pattern; must appear once.
    monkeypatch.setattr("os.environ", {
        "OPENAI_API_KEY": "sk-openaiSECRET789",
    }, raising=False)
    r = _run(EnvCredentialProbe().probe())
    assert r.env_keys == ["OPENAI_API_KEY"]


def test_env_probe_empty(monkeypatch):
    monkeypatch.setattr("os.environ", {"HOME": "/home/user"}, raising=False)
    r = _run(EnvCredentialProbe().probe())
    assert r.available is False
    assert r.env_keys == []
    d = r.to_dict()
    # env_keys omitted when empty
    assert "env_keys" not in d


def test_env_probe_values_never_serialized(monkeypatch):
    secret_val = "sk-ant-DEADBEEFdeadbeef99887766"
    monkeypatch.setattr("os.environ", {
        "ANTHROPIC_API_KEY": secret_val,
        "STRIPE_SECRET_KEY": "sk_live_ANOTHERSECRETvalue",
    }, raising=False)
    r = _run(EnvCredentialProbe().probe())
    blob = json.dumps(r.to_dict())
    assert secret_val not in blob
    assert "ANOTHERSECRETvalue" not in blob
    # names ARE present
    assert "ANTHROPIC_API_KEY" in blob
    # no fragment of a value: check a distinctive substring
    assert "DEADBEEF" not in blob


# ── HostDiscovery cache round-trip ────────────────────────────────────

def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("os.environ", {
        "ANTHROPIC_API_KEY": "sk-ant-CACHEDSECRET123456",
    }, raising=False)
    hd = HostDiscovery(tmp_path)
    # Scan only the env-credential probe to keep it fast and deterministic.
    _run(hd.scan(probe_ids=["host_env_credentials"]))

    cache = tmp_path / "security" / "host_scan.json"
    assert cache.exists()
    # No value leaked to disk.
    assert "CACHEDSECRET" not in cache.read_text(encoding="utf-8")

    env_result = next(r for r in hd.results if r.id == "host_env_credentials")
    assert env_result.env_keys == ["ANTHROPIC_API_KEY"]
    # importable == the found names for the env probe.
    assert env_result.importable == ["ANTHROPIC_API_KEY"]

    # Reload from cache in a fresh instance.
    hd2 = HostDiscovery(tmp_path)
    assert hd2.has_scanned
    assert hd2.last_scan == hd.last_scan
    reloaded = next(r for r in hd2.results if r.id == "host_env_credentials")
    assert reloaded.env_keys == ["ANTHROPIC_API_KEY"]
    assert reloaded.importable == ["ANTHROPIC_API_KEY"]


def test_cache_absent_is_clean(tmp_path):
    hd = HostDiscovery(tmp_path)
    assert hd.has_scanned is False
    assert hd.results == []


def test_importable_for_keyed_probe(tmp_path, monkeypatch):
    # A keyed probe (host_anthropic) should carry ANTHROPIC_API_KEY as
    # importable when the env var is present.
    monkeypatch.setattr("os.environ", {
        "ANTHROPIC_API_KEY": "sk-ant-KEYEDPROBE0001",
    }, raising=False)
    hd = HostDiscovery(tmp_path)
    _run(hd.scan(probe_ids=["host_anthropic"]))
    r = next(x for x in hd.results if x.id == "host_anthropic")
    assert r.importable == ["ANTHROPIC_API_KEY"]
    # The anthropic probe's own detail truncates the key; ensure the FULL
    # value never lands in the serialized dict.
    assert "KEYEDPROBE0001" not in json.dumps(r.to_dict())


# ── secrets_import name validation ────────────────────────────────────

@pytest.mark.parametrize("name,expected_ok", [
    ("ANTHROPIC_API_KEY", True),
    ("my_service", True),
    ("", False),
    ("has space", False),
    ("has,comma", False),
    ("none", False),
    ("NONE", False),
    ("all", False),
    ("ALL", False),
])
def test_validate_secret_name(name, expected_ok):
    reason = validate_secret_name(name)
    assert (reason is None) == expected_ok
    if not expected_ok:
        assert isinstance(reason, str) and reason
