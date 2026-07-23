"""Vault-backed credential migration + daemon-plane grant isolation.

Covers the migration of daemon-held credentials into the age-encrypted vault:
  - CredentialStore backend swap (vault primary, NO plaintext fallback)
  - legacy credentials/*.json migration into the vault
  - agent wallet keys out of identity.json into agent-key.<id>
  - dotted (daemon-plane) names are never agent-grantable

All secret values used here are DUMMY fixtures. The vault lives under a
tmp_path KEYSTORE_DIR so nothing touches the real keystore.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("pyrage", reason="vault keystore needs pyrage")

from atn._vendor.kevin import keystore


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    """Point the keystore module at a fresh tmp KEYSTORE_DIR.

    KEYSTORE_DIR is read at import time into module-level path constants, so we
    override those constants directly (env alone would not take effect on an
    already-imported module)."""
    ks_dir = tmp_path / "keystore"
    ks_dir.mkdir()
    monkeypatch.setenv("KEYSTORE_DIR", str(ks_dir))
    monkeypatch.setattr(keystore, "KEYSTORE_DIR", str(ks_dir))
    monkeypatch.setattr(keystore, "KEY_PATH", str(ks_dir / "identity.age-key"))
    monkeypatch.setattr(keystore, "VAULT_PATH", str(ks_dir / "vault.age"))
    monkeypatch.setattr(keystore, "BUNDLES_PATH", str(ks_dir / "bundles.json"))
    monkeypatch.setattr(keystore, "EXPORT_DIR", str(ks_dir / "exports"))
    keystore.init_identity()
    return keystore


# ── Task 1: CredentialStore round-trip + no plaintext ─────────────────────────

def test_credential_store_roundtrip_no_plaintext(vault, tmp_path):
    from atn.credentials import CredentialStore, PREFIX

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = CredentialStore(data_dir)

    assert store.exists("google_calendar") is False
    assert store.load("google_calendar") == {}
    assert store.list_integrations() == []

    creds = {"access_token": "DUMMY_AT", "refresh_token": "DUMMY_RT"}
    store.save("google_calendar", creds)

    assert store.exists("google_calendar") is True
    assert store.load("google_calendar") == creds
    assert "google_calendar" in store.list_integrations()
    # The entry lives in the vault under the dotted app.* prefix.
    assert (PREFIX + "google_calendar") in vault.list_services()

    # No plaintext credential file was written.
    cred_files = list((data_dir / "credentials").glob("*.json")) \
        if (data_dir / "credentials").exists() else []
    assert cred_files == [], "CredentialStore must not write plaintext files"

    store.delete("google_calendar")
    assert store.exists("google_calendar") is False
    assert store.load("google_calendar") == {}


# ── Task 1: legacy file migration ─────────────────────────────────────────────

def test_legacy_file_migration(vault, tmp_path):
    from atn.credentials import CredentialStore, PREFIX

    data_dir = tmp_path / "data"
    cred_dir = data_dir / "credentials"
    cred_dir.mkdir(parents=True)
    legacy = {"api_key": "DUMMY_LEGACY_KEY"}
    legacy_file = cred_dir / "slack.json"
    legacy_file.write_text(json.dumps(legacy), encoding="utf-8")

    # Constructing the store triggers one-shot migration.
    store = CredentialStore(data_dir)

    assert (PREFIX + "slack") in vault.list_services()
    assert not legacy_file.exists(), "legacy plaintext file must be deleted"
    assert store.load("slack") == legacy
    assert "slack" in store.list_integrations()


def test_legacy_file_deleted_when_vault_already_has_it(vault, tmp_path):
    from atn.credentials import CredentialStore, PREFIX

    data_dir = tmp_path / "data"
    cred_dir = data_dir / "credentials"
    cred_dir.mkdir(parents=True)
    # Vault already holds the entry with a DIFFERENT value.
    vault.put_secret(PREFIX + "notion", json.dumps({"api_key": "DUMMY_VAULT"}))
    (cred_dir / "notion.json").write_text(
        json.dumps({"api_key": "DUMMY_STALE_FILE"}), encoding="utf-8")

    store = CredentialStore(data_dir)

    # File gone; the vault value (not the stale file) is authoritative.
    assert not (cred_dir / "notion.json").exists()
    assert store.load("notion") == {"api_key": "DUMMY_VAULT"}


# ── Task 2: agent wallet keys out of identity.json ────────────────────────────

DUMMY_KEY = "0x" + "11" * 32

# Derive the matching address when eth_account is present so _load_identity's
# key/address consistency check passes; fall back to a placeholder otherwise.
try:
    from eth_account import Account as _Account
    DUMMY_ADDR = _Account.from_key(DUMMY_KEY).address
except Exception:  # pragma: no cover - offline/local-only
    DUMMY_ADDR = "0xabc0000000000000000000000000000000000abc"


class _FakeIdentity:
    def __init__(self):
        self.public_key = "0xpub"
        self.address = DUMMY_ADDR
        self.lineage_hash = "0xlin"
        self.registered_on_chain = False
        self.registration_tx = None


def _make_registry(vault, tmp_path):
    """Build a bare AgentRegistry-like object exposing the identity methods,
    without constructing the full runtime graph. We only need the static
    keystore helpers + the _save_identity/_load_identity code paths, so we bind
    a minimal config and monkeypatch the keystore import."""
    from atn.runtime import agent_registry as ar

    # Point the registry's keystore import at our tmp vault.
    def _fake_keystore():
        return vault
    ar.AgentRegistry._keystore = staticmethod(_fake_keystore)

    class _Cfg:
        agents_dir = tmp_path / "agents"

    reg = ar.AgentRegistry.__new__(ar.AgentRegistry)
    reg._config = _Cfg()
    reg._agent_keys = {}
    return reg


def test_save_identity_key_goes_to_vault_not_file(vault, tmp_path):
    reg = _make_registry(vault, tmp_path)
    ident = _FakeIdentity()

    reg._save_identity("agent1", ident, DUMMY_KEY)

    path = reg._identity_path("agent1")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "private_key" not in data, "identity.json must not contain the key"
    assert data["address"] == ident.address
    assert ("agent-key.agent1") in vault.list_services()


def test_load_identity_reads_key_from_vault(vault, tmp_path):
    reg = _make_registry(vault, tmp_path)
    ident = _FakeIdentity()
    reg._save_identity("agent2", ident, DUMMY_KEY)

    # eth_account may or may not be installed; either way the key should load.
    loaded = reg._load_identity("agent2")
    assert loaded is not None
    _identity, key = loaded
    assert key == DUMMY_KEY


def test_legacy_identity_migrates_key_to_vault(vault, tmp_path):
    reg = _make_registry(vault, tmp_path)
    # Pre-place a legacy identity.json WITH a private_key field.
    path = reg._identity_path("agent3")
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = {
        "public_key": "0xpub",
        "address": DUMMY_ADDR,
        "lineage_hash": "0xlin",
        "private_key": DUMMY_KEY,
        "registered_on_chain": False,
        "registration_tx": None,
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = reg._load_identity("agent3")
    assert loaded is not None
    _identity, key = loaded
    assert key == DUMMY_KEY
    # Vault now holds the key; the file no longer does.
    assert "agent-key.agent3" in vault.list_services()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "private_key" not in on_disk


# ── Task 3: grantability isolation ────────────────────────────────────────────

def test_generate_policy_map_excludes_dotted(vault):
    from atn.vault_setup import generate_policy_map

    vault.put_secret("GITHUB_TOKEN", "DUMMY")          # flat -> grantable
    vault.put_secret("app.google_calendar", "{}")      # dotted -> excluded
    vault.put_secret("agent-key.agent1", "DUMMY")      # dotted -> excluded

    mapping = generate_policy_map(vault)
    assert "GITHUB_TOKEN" in mapping
    assert "app.google_calendar" not in mapping
    assert "agent-key.agent1" not in mapping
    assert all("." not in k for k in mapping)


def test_validate_secret_name_rejects_dotted():
    from atn.runtime.host_discovery import validate_secret_name

    assert validate_secret_name("GITHUB_TOKEN") is None
    assert validate_secret_name("app.google_calendar") is not None
    assert validate_secret_name("agent-key.agent1") is not None
    # Existing rules still hold.
    assert validate_secret_name("") is not None
    assert validate_secret_name("all") is not None
    assert validate_secret_name("a,b") is not None


def test_worker_resolve_spec_filters_dotted(vault, monkeypatch):
    from atn.runtime import worker_host

    vault.put_secret("GITHUB_TOKEN", "DUMMY")
    vault.put_secret("app.google_calendar", "{}")
    vault.put_secret("agent-key.agent1", "DUMMY")

    # worker_host imported its own resolve_spec at module load; point the raw
    # resolver at our tmp vault so the filter runs over real vault contents.
    monkeypatch.setattr(worker_host, "_raw_resolve_spec", vault.resolve_spec)

    resolved_all = worker_host._resolve_spec("all")
    assert "GITHUB_TOKEN" in resolved_all
    assert "app.google_calendar" not in resolved_all
    assert "agent-key.agent1" not in resolved_all

    # Named literally, a dotted service still does not resolve.
    resolved_named = worker_host._resolve_spec("app.google_calendar,GITHUB_TOKEN")
    assert resolved_named == ["GITHUB_TOKEN"]
