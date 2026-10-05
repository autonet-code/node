"""Copy compiled contract ABIs into the packaged ABI dir.

``artifacts/`` is gitignored and never reaches a pip install, so the
runtime ships ABI-only copies at ``nodes/common/abi/<Name>.json`` (see
``nodes/common/contract_artifacts.py``). Run after ``npx hardhat compile``
whenever a contract interface changes:

    python scripts/sync_contract_abis.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nodes.common.contract_artifacts import (  # noqa: E402
    PACKAGED_ABI_DIR,
    REPO_ARTIFACTS_DIR,
    _hardhat_relpath,
)

# Contracts the Python runtime loads by artifact. Extend as callers grow.
CONTRACTS = ["Substrate"]


def main() -> int:
    PACKAGED_ABI_DIR.mkdir(parents=True, exist_ok=True)
    for name in CONTRACTS:
        src = REPO_ARTIFACTS_DIR / _hardhat_relpath(name)
        if not src.is_file():
            print(f"missing {src}; run `npx hardhat compile` first")
            return 1
        abi = json.loads(src.read_text(encoding="utf-8"))["abi"]
        dst = PACKAGED_ABI_DIR / f"{name}.json"
        dst.write_text(
            json.dumps({"contractName": name, "abi": abi}, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {dst.relative_to(REPO)} ({len(abi)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
