"""Contract ABI resolution shared by every chain-facing module.

Replaces hardcoded absolute ``artifacts/`` lookups that only
worked on one developer machine. Resolution order for a contract's
artifact JSON (anything with an ``"abi"`` key):

1. ``ATN_ARTIFACTS_DIR`` — an explicit Hardhat ``artifacts/`` root
   (``<dir>/contracts/core/<Name>.sol/<Name>.json``). Set by operators
   and containers that mount freshly compiled artifacts.
2. The repo-relative Hardhat ``artifacts/`` dir (source checkouts after
   ``npx hardhat compile``). A fresh compile wins over the packaged copy.
3. The packaged ABI-only copy at ``nodes/common/abi/<Name>.json``,
   shipped as package data so ``pip install autonet-computer`` works
   without Hardhat. Refresh it with ``scripts/sync_contract_abis.py``;
   ``tests/test_contract_artifacts.py`` fails when it drifts from a
   locally compiled artifact.

Only the ABI is guaranteed: the packaged copy omits bytecode.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ARTIFACTS_ENV = "ATN_ARTIFACTS_DIR"

# nodes/common/contract_artifacts.py -> repo root is two parents up from
# the package dir. In a pip install this points into site-packages, where
# no artifacts/ dir exists, and resolution falls through to package data.
_PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGED_ABI_DIR = _PACKAGE_DIR / "abi"
REPO_ARTIFACTS_DIR = _PACKAGE_DIR.parent.parent / "artifacts"

# Contract name -> source subdir under contracts/ (Hardhat layout).
_CONTRACT_SOURCES = {
    "Substrate": "core",
    "ServiceMarket": "core",
    "VentureVault": "core",
    "CharterAnchor": "core",
}


def _hardhat_relpath(name: str) -> Path:
    sub = _CONTRACT_SOURCES.get(name, "core")
    return Path("contracts") / sub / f"{name}.sol" / f"{name}.json"


def artifact_candidates(name: str) -> List[Path]:
    """Ordered candidate paths for ``name``'s artifact JSON."""
    rel = _hardhat_relpath(name)
    out: List[Path] = []
    env_dir = os.environ.get(ARTIFACTS_ENV, "").strip()
    if env_dir:
        out.append(Path(env_dir).expanduser() / rel)
    out.append(REPO_ARTIFACTS_DIR / rel)
    out.append(PACKAGED_ABI_DIR / f"{name}.json")
    return out


def resolve_artifact_path(name: str) -> Path:
    """First existing candidate path for ``name``.

    Raises FileNotFoundError listing every place searched.
    """
    candidates = artifact_candidates(name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no ABI for contract {name!r}; searched: "
        + ", ".join(str(p) for p in candidates)
    )


def load_artifact(name: str) -> Dict[str, Any]:
    """Load ``name``'s artifact JSON (at least ``{"abi": [...]}``)."""
    path = resolve_artifact_path(name)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not isinstance(data.get("abi"), list):
        raise ValueError(f"artifact {path} has no 'abi' list")
    return data


def load_abi(name: str) -> List[Dict[str, Any]]:
    """Load ``name``'s ABI list."""
    return load_artifact(name)["abi"]
