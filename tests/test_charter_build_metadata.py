"""The default charter exists twice: in this repo (DEFAULT_CHARTER) and in
trustless-contracts' published Autonet module build metadata, which werule
passes at install (dao, werule 6d28c2d). Nothing else checks that the two
agree, so this test does whenever that repo is on disk next to this one."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nodes.common.library.charter import DEFAULT_CHARTER

DAO_DIR = Path(os.environ.get("TRUSTLESS_CONTRACTS_DIR",
                              r"C:/code/dao/trustless-contracts"))
META = DAO_DIR / "modules" / "autonet" / "build-metadata.json"


def _charter_default(meta: dict) -> str:
    inputs = meta["pluginSetup"]["prepareInstallation"]["inputs"]
    for inp in inputs:
        for comp in inp.get("components") or []:
            if comp.get("name") == "charter":
                return comp["x-ui"]["default"]
    raise AssertionError("no charter input in the build metadata")


@pytest.mark.skipif(not META.is_file(), reason="trustless-contracts not on disk")
def test_build_metadata_default_is_the_daemons_default_charter():
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert _charter_default(meta) == DEFAULT_CHARTER.text
