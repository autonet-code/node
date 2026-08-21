"""User dossier (USER.md) store, profile tool pair, and first-boot seeding."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from atn.user_profile import UserProfileStore
from atn.agent_tools import (
    _get_user_profile,
    _update_user_profile,
    _TOOL_CATEGORIES,
    resolve_tool_surface,
)
from atn.fleet_seed import seed_default_fleet, KEVIN_ID


SAMPLE = """\
# Jane Doe

Engineer. Based in Lisbon.

## Skills

- Rust (as of Aug 2026)
- Sailing

## Goals

Ship the boat project.
"""


@pytest.fixture
def store(tmp_path: Path) -> UserProfileStore:
    s = UserProfileStore(tmp_path)
    s.dossier_path.write_text(SAMPLE, encoding="utf-8")
    return s


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_sections_and_read(store: UserProfileStore):
    assert store.dossier_sections() == ["Skills", "Goals"]
    assert "Rust" in store.read_dossier_section("skills")  # case-insensitive
    assert store.read_dossier_section("nope") is None


def test_replace_preserves_preamble_and_others(store: UserProfileStore):
    store.write_dossier_section("Skills", "- Zig only now")
    text = store.read_dossier()
    assert "# Jane Doe" in text and "Lisbon" in text          # preamble kept
    assert "Zig only now" in text and "Rust" not in text      # replaced
    assert "Ship the boat project." in text                   # sibling kept


def test_append_and_create(store: UserProfileStore):
    store.write_dossier_section("Goals", "Learn celestial navigation.", mode="append")
    assert "boat project" in store.read_dossier_section("Goals")
    assert "celestial" in store.read_dossier_section("Goals")

    res = store.write_dossier_section("Constraints", "Limited weekends.")
    assert res["existed"] is False
    assert store.dossier_sections() == ["Skills", "Goals", "Constraints"]


def test_remove(store: UserProfileStore):
    store.write_dossier_section("Skills", mode="remove")
    assert store.dossier_sections() == ["Goals"]
    # removing a missing section is a tolerated no-op
    res = store.write_dossier_section("Skills", mode="remove")
    assert res["existed"] is False


def test_empty_dossier(tmp_path: Path):
    s = UserProfileStore(tmp_path)
    assert s.read_dossier() == ""
    assert s.dossier_sections() == []
    s.write_dossier_section("Skills", "- Python")
    assert s.read_dossier_section("Skills") == "- Python"


def test_bad_args(store: UserProfileStore):
    with pytest.raises(ValueError):
        store.write_dossier_section("", "x")
    with pytest.raises(ValueError):
        store.write_dossier_section("Skills", "x", mode="explode")


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _fake_runtime(store: UserProfileStore):
    return SimpleNamespace(user_profile=store, list_agents=lambda: ["a", "b"])


def test_get_user_profile_full(store: UserProfileStore):
    out = asyncio.run(_get_user_profile(_fake_runtime(store), {}))
    assert "Jane Doe" in out["dossier"]
    assert out["sections"] == ["Skills", "Goals"]
    assert out["goal_count"] == 2
    assert "onboarding_status" not in out


def test_get_user_profile_section(store: UserProfileStore):
    out = asyncio.run(_get_user_profile(_fake_runtime(store), {"section": "skills"}))
    assert "Rust" in out["content"]
    missing = asyncio.run(_get_user_profile(_fake_runtime(store), {"section": "zzz"}))
    assert "error" in missing and missing["sections"] == ["Skills", "Goals"]


def test_update_user_profile(store: UserProfileStore):
    rt = _fake_runtime(store)
    out = asyncio.run(_update_user_profile(
        rt, {"section": "Skills", "content": "- Zig", "mode": "replace"}))
    assert out["existed"] is True
    assert store.read_dossier_section("Skills") == "- Zig"

    assert "error" in asyncio.run(_update_user_profile(rt, {"section": ""}))
    big = "x" * 20_001
    assert "error" in asyncio.run(
        _update_user_profile(rt, {"section": "Skills", "content": big}))


def test_profile_category_bundle():
    assert _TOOL_CATEGORIES["profile"] == {"get_user_profile", "update_user_profile"}
    names = {t["name"] for t in resolve_tool_surface(["profile"])}
    assert names == {"get_user_profile", "update_user_profile"}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _seed_env(tmp_path: Path, agents: list | None = None):
    registered: list = []

    async def register_agent(defn, *, legacy=False):
        registered.append((defn, legacy))
        return defn.id

    runtime = SimpleNamespace(
        list_agents=lambda: list(agents or []),
        register_agent=register_agent,
    )
    config = SimpleNamespace(data_dir=tmp_path, default_provider="claude_max",
                             default_model="claude-fable-5")
    return runtime, config, registered


def test_seed_fresh_install(tmp_path: Path):
    runtime, config, registered = _seed_env(tmp_path)
    assert asyncio.run(seed_default_fleet(runtime, config)) == KEVIN_ID
    (defn, legacy) = registered[0]
    assert defn.id == KEVIN_ID and legacy is True
    assert defn.cognitive_model == "claude-fable-5"
    assert "profile" in defn.tools and "sdk_builtin" in defn.tools
    assert (tmp_path / ".fleet_seeded").exists()
    # Second boot: stamp short-circuits, even with zero agents (Kevin removed).
    assert asyncio.run(seed_default_fleet(runtime, config)) is None
    assert len(registered) == 1


def test_seed_existing_fleet_stamps_and_skips(tmp_path: Path):
    runtime, config, registered = _seed_env(tmp_path, agents=["existing"])
    assert asyncio.run(seed_default_fleet(runtime, config)) is None
    assert registered == []
    assert (tmp_path / ".fleet_seeded").exists()
