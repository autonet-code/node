"""Contract ABI resolver (nodes/common/contract_artifacts.py).

Order: ATN_ARTIFACTS_DIR -> repo artifacts/ -> packaged ABI copy.
"""

from __future__ import annotations

import json

import pytest

from nodes.common import contract_artifacts as ca


def _write_hardhat(root, name, abi):
    p = root / "contracts" / "core" / f"{name}.sol" / f"{name}.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"abi": abi, "bytecode": "0x"}), encoding="utf-8")
    return p


def test_env_override_wins(tmp_path, monkeypatch):
    abi = [{"type": "function", "name": "envOnly"}]
    p = _write_hardhat(tmp_path, "Substrate", abi)
    monkeypatch.setenv(ca.ARTIFACTS_ENV, str(tmp_path))
    assert ca.resolve_artifact_path("Substrate") == p
    assert ca.load_abi("Substrate") == abi


def test_falls_back_to_packaged_abi(tmp_path, monkeypatch):
    # Simulate a pip install: no env, no repo artifacts dir.
    monkeypatch.delenv(ca.ARTIFACTS_ENV, raising=False)
    monkeypatch.setattr(ca, "REPO_ARTIFACTS_DIR", tmp_path / "nope")
    path = ca.resolve_artifact_path("Substrate")
    assert path == ca.PACKAGED_ABI_DIR / "Substrate.json"
    names = {e.get("name") for e in ca.load_abi("Substrate")}
    # Functions the four former hardcoded call sites rely on.
    assert {"submitAnchor", "recordTrainingForEpoch", "anchorCount",
            "getAnchor"} <= names


def test_missing_everywhere_raises_with_search_list(tmp_path, monkeypatch):
    monkeypatch.setenv(ca.ARTIFACTS_ENV, str(tmp_path / "env"))
    monkeypatch.setattr(ca, "REPO_ARTIFACTS_DIR", tmp_path / "repo")
    monkeypatch.setattr(ca, "PACKAGED_ABI_DIR", tmp_path / "pkg")
    with pytest.raises(FileNotFoundError) as ei:
        ca.load_abi("Substrate")
    msg = str(ei.value)
    assert "env" in msg and "repo" in msg and "pkg" in msg


def test_rejects_artifact_without_abi(tmp_path, monkeypatch):
    p = tmp_path / "contracts" / "core" / "Substrate.sol" / "Substrate.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(ca.ARTIFACTS_ENV, str(tmp_path))
    with pytest.raises(ValueError):
        ca.load_abi("Substrate")


def test_packaged_abi_matches_compiled_artifact():
    """Drift guard: the shipped copy must equal a local compile.
    Refresh with scripts/sync_contract_abis.py."""
    compiled = (ca.REPO_ARTIFACTS_DIR / "contracts" / "core"
                / "Substrate.sol" / "Substrate.json")
    if not compiled.is_file():
        pytest.skip("no local hardhat compile")
    packaged = ca.PACKAGED_ABI_DIR / "Substrate.json"
    a = json.loads(compiled.read_text(encoding="utf-8"))["abi"]
    b = json.loads(packaged.read_text(encoding="utf-8"))["abi"]
    assert a == b, "packaged ABI drifted; run scripts/sync_contract_abis.py"


def test_hardcoded_paths_gone():
    """No runtime module may hardcode a developer-machine artifacts path."""
    from pathlib import Path
    root = Path(ca.__file__).resolve().parents[2]
    offenders = []
    for sub in ("atn", "nodes"):
        for py in (root / sub).rglob("*.py"):
            if "C:/code/autonet/artifacts" in py.read_text(
                    encoding="utf-8", errors="ignore"):
                offenders.append(str(py.relative_to(root)))
    assert offenders == []
