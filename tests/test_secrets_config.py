"""security-settings surface: save_secrets_config_to_yaml round-trip,
secrets_config allowance-spec validation, and read-only GET shape.

These are unit-level tests over the config save function and the WS handler's
validation/response shape. They do NOT touch the live :7700 daemon.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import yaml

from atn.config import save_secrets_config_to_yaml


# ── yaml round-trip of save_secrets_config_to_yaml ────────────────────────────

def test_save_creates_both_sections(tmp_path):
    cfg = tmp_path / "config.yaml"
    save_secrets_config_to_yaml(
        worker_isolation=True,
        default_root_allowance="all",
        config_path=cfg,
    )
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert raw["worker_isolation"]["enabled"] is True
    assert raw["secrets"]["default_root_allowance"] == "all"


def test_save_partial_updates_only_passed_keys(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({
        "worker_isolation": {"enabled": False, "memory_cap_mb": 512},
        "secrets": {"default_root_allowance": "none"},
        "data_dir": "/somewhere",
    }), encoding="utf-8")

    # Only flip isolation; leave allowance + unrelated keys alone.
    save_secrets_config_to_yaml(worker_isolation=True, config_path=cfg)

    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert raw["worker_isolation"]["enabled"] is True
    assert raw["worker_isolation"]["memory_cap_mb"] == 512  # preserved
    assert raw["secrets"]["default_root_allowance"] == "none"  # untouched
    assert raw["data_dir"] == "/somewhere"  # untouched


def test_save_allowance_only_leaves_isolation_absent(tmp_path):
    cfg = tmp_path / "config.yaml"
    save_secrets_config_to_yaml(
        default_root_allowance="github,slack", config_path=cfg)
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert raw["secrets"]["default_root_allowance"] == "github,slack"
    assert "worker_isolation" not in raw


def test_save_roundtrips_through_load_config(tmp_path, monkeypatch):
    from atn import config as config_mod

    cfg = tmp_path / "config.yaml"
    # No ATN_WORKER_ISOLATION env override for a clean file->config round-trip.
    monkeypatch.delenv("ATN_WORKER_ISOLATION", raising=False)
    save_secrets_config_to_yaml(
        worker_isolation=True,
        default_root_allowance="github",
        config_path=cfg,
    )
    loaded = config_mod.load_config(cfg)
    assert loaded.worker_isolation.enabled is True
    assert loaded.secrets.default_root_allowance == "github"


# ── WS handler: validation + response shape ───────────────────────────────────

def _make_bridge():
    """A minimal WebSocketBridge stub exposing only what _handle_secrets_config
    reads: runtime._config (worker_isolation / secrets) + runtime._broker_client.
    Persistence is redirected to a tmp config via monkeypatch on the caller."""
    from atn.ws_server import WebSocketBridge

    runtime = SimpleNamespace(
        _config=SimpleNamespace(
            worker_isolation=SimpleNamespace(enabled=False, memory_cap_mb=0),
            secrets=SimpleNamespace(default_root_allowance="none"),
        ),
        _broker_client=SimpleNamespace(value_push_armed=False),
    )
    bridge = WebSocketBridge.__new__(WebSocketBridge)
    bridge.runtime = runtime
    return bridge


def _call(bridge, msg):
    return asyncio.run(bridge._handle_secrets_config(msg, msg_id=7))


def test_readonly_get_shape(monkeypatch):
    bridge = _make_bridge()
    bridge.runtime._config.worker_isolation.enabled = True
    bridge.runtime._config.secrets.default_root_allowance = "all"

    resp = _call(bridge, {})  # empty payload -> GET
    assert resp["ok"] is True
    r = resp["result"]
    assert set(r) == {"worker_isolation", "default_root_allowance",
                      "isolation_live", "restart_required", "warning"}
    assert r["worker_isolation"] is True
    assert r["default_root_allowance"] == "all"
    assert r["isolation_live"] is True
    assert r["restart_required"] is False  # no change requested
    assert r["warning"] is None


def test_allowance_rejects_dotted(monkeypatch):
    bridge = _make_bridge()
    resp = _call(bridge, {"default_root_allowance": "github,app.google_calendar"})
    assert resp["ok"] is False
    assert "dotted" in resp["error"].lower()


def test_allowance_accepts_flat_and_persists(tmp_path, monkeypatch):
    import atn.config as config_mod
    cfg = tmp_path / "config.yaml"
    monkeypatch.setattr(config_mod, "_DEFAULT_DIR", tmp_path)

    bridge = _make_bridge()
    resp = _call(bridge, {"default_root_allowance": "github, slack"})
    assert resp["ok"] is True
    assert resp["result"]["default_root_allowance"] == "github, slack"
    # In-memory applied.
    assert bridge.runtime._config.secrets.default_root_allowance == "github, slack"
    # Persisted.
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert raw["secrets"]["default_root_allowance"] == "github, slack"


def test_enable_isolation_warns_broker_down(tmp_path, monkeypatch):
    import atn.config as config_mod
    monkeypatch.setattr(config_mod, "_DEFAULT_DIR", tmp_path)
    monkeypatch.delenv("ATN_WORKER_ISOLATION", raising=False)

    bridge = _make_bridge()
    bridge.runtime._broker_client.value_push_armed = False
    resp = _call(bridge, {"worker_isolation": True})
    assert resp["ok"] is True
    r = resp["result"]
    assert r["worker_isolation"] is True
    assert r["restart_required"] is True  # orphan monitor loop
    assert r["warning"] is not None
    assert "broker" in r["warning"].lower()
    assert "atn-vault-setup" in r["warning"]


def test_enable_isolation_env_override_warning(tmp_path, monkeypatch):
    import atn.config as config_mod
    monkeypatch.setattr(config_mod, "_DEFAULT_DIR", tmp_path)
    monkeypatch.setenv("ATN_WORKER_ISOLATION", "0")

    bridge = _make_bridge()
    bridge.runtime._broker_client.value_push_armed = True
    resp = _call(bridge, {"worker_isolation": True})
    assert resp["ok"] is True
    assert "ATN_WORKER_ISOLATION" in resp["result"]["warning"]


def test_allowance_blank_normalizes_to_none(tmp_path, monkeypatch):
    import atn.config as config_mod
    monkeypatch.setattr(config_mod, "_DEFAULT_DIR", tmp_path)

    bridge = _make_bridge()
    resp = _call(bridge, {"default_root_allowance": "   "})
    assert resp["ok"] is True
    assert resp["result"]["default_root_allowance"] == "none"
