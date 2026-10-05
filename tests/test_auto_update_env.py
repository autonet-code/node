"""ATN_AUTO_UPDATE env override (the daemon container image sets it to 0)."""
from atn import config as config_mod


def test_env_disables_without_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ATN_AUTO_UPDATE", "0")
    cfg = config_mod.load_config(tmp_path / "missing.yaml")
    assert cfg.auto_update.enabled is False


def test_env_wins_over_config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("auto_update:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("ATN_AUTO_UPDATE", "off")
    assert config_mod.load_config(path).auto_update.enabled is False
    monkeypatch.setenv("ATN_AUTO_UPDATE", "1")
    path.write_text("auto_update:\n  enabled: false\n", encoding="utf-8")
    assert config_mod.load_config(path).auto_update.enabled is True


def test_unset_keeps_config_value(tmp_path, monkeypatch):
    monkeypatch.delenv("ATN_AUTO_UPDATE", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("auto_update:\n  enabled: false\n", encoding="utf-8")
    assert config_mod.load_config(path).auto_update.enabled is False
    assert config_mod.load_config(tmp_path / "missing.yaml").auto_update.enabled is True
