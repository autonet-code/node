"""The federated close must anchor to the Substrate atn registers against.

nodes.common.config used to prefer a registry.json next to the config file
(~/.atn/registry.json) over the copy packaged with atn, so a stale copy from an
older release pointed anchoring at a retired Substrate after a redeploy.
"""
import json
from pathlib import Path

from nodes.common.config import _registry_substrate_address

REPO = Path(__file__).resolve().parents[1]


def _packaged_substrate() -> str:
    data = json.loads((REPO / "atn" / "registry.json").read_text(encoding="utf-8"))
    return next(
        e["contracts"]["substrate"]
        for e in data["jurisdictions"].values()
        if e.get("contracts", {}).get("substrate")
    )


def test_packaged_registry_beats_stale_copy_next_to_config(tmp_path):
    stale = {"jurisdictions": {"autonet": {"contracts": {
        "substrate": "0x000000000000000000000000000000000000dEaD"}}}}
    (tmp_path / "registry.json").write_text(json.dumps(stale), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("", encoding="utf-8")

    assert _registry_substrate_address(str(config_path)) == _packaged_substrate()


def test_repo_root_registry_matches_packaged_copy():
    root = json.loads((REPO / "registry.json").read_text(encoding="utf-8"))
    packaged = json.loads((REPO / "atn" / "registry.json").read_text(encoding="utf-8"))
    assert root == packaged


def test_rep_token_reaches_the_close_config(tmp_path):
    # Without it the close runs in the genesis regime: every review weighs
    # the same and voice weights are all epsilon.
    from nodes.common.config import load_config
    data = json.loads((REPO / "atn" / "registry.json").read_text(encoding="utf-8"))
    rep = next(e["contracts"]["rep_token"] for e in data["jurisdictions"].values()
               if e.get("contracts", {}).get("rep_token"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("", encoding="utf-8")
    assert load_config(str(config_path)).blockchain.rep_token_address == rep


def test_chosen_jurisdiction_skips_the_packaged_fallback(tmp_path):
    # A chosen jurisdiction's addresses come from the atn side; the packaged
    # (Shadownet) ones must not fill the gaps.
    from nodes.common.config import load_config
    config_path = tmp_path / "config.yaml"
    config_path.write_text("", encoding="utf-8")
    bc = load_config(str(config_path), registry_fallback=False).blockchain
    assert not bc.substrate_address and not bc.rep_token_address
    assert not bc.governor_address and not bc.deploy_block
