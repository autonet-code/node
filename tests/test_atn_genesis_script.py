"""scripts/build_atn_genesis.py end to end on eth_tester: snapshot an old
Substrate's ATN balances, drop exclusions, deploy a new Substrate with the
root, and claim every leaf with the emitted proofs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("eth_tester")

from web3 import Web3  # noqa: E402
from web3.providers.eth_tester import EthereumTesterProvider  # noqa: E402

from tests.substrate_chain_helpers import SLOT_VAULT, deploy_substrate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "artifacts" / "contracts" / "core" / "Substrate.sol" / "Substrate.json"

_spec = importlib.util.spec_from_file_location(
    "build_atn_genesis", REPO / "scripts" / "build_atn_genesis.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_snapshot_exclude_and_claim(tmp_path):
    if not ARTIFACT.exists():
        pytest.skip(f"missing artifact: {ARTIFACT}")
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    w3 = Web3(EthereumTesterProvider())
    acc = w3.eth.accounts
    vault = acc[SLOT_VAULT]
    old = w3.eth.contract(
        address=deploy_substrate(w3, art["abi"], art["bytecode"], acc[0]),
        abi=art["abi"])

    # Balances: a=600, b=300 (after a transfer), c=100 (excluded), d=0.
    for to, amt in ((acc[1], 1000), (acc[3], 100), (acc[4], 50)):
        old.functions.mintFromVault(to, amt).transact({"from": vault})
    old.functions.transfer(acc[2], 300).transact({"from": acc[1]})
    old.functions.transfer(acc[1], 50).transact({"from": acc[4]})
    old.functions.transfer(acc[2], 150).transact({"from": acc[1]})
    snap = w3.eth.block_number
    old.functions.mintFromVault(acc[1], 999).transact({"from": vault})  # after snapshot

    excl = tmp_path / "excl.txt"
    excl.write_text(f"# treasury\n{acc[3]}\n", encoding="utf-8")
    bal = gen.atn_balances(w3, old.address, snap, chunk=2)
    entries = gen.filter_balances(bal, gen.load_exclusions(str(excl)))
    assert dict(entries) == {acc[1]: 600, acc[2]: 450}
    root, claims = gen.build_tree(entries)

    new = w3.eth.contract(
        address=deploy_substrate(w3, art["abi"], art["bytecode"], acc[0],
                                 genesis_root=root),
        abi=art["abi"])
    for addr, c in claims.items():
        new.functions.claimGenesis(int(c["amount"]), c["proof"]).transact({"from": addr})
        assert new.functions.balanceOf(addr).call() == int(c["amount"])
    assert new.functions.genesisMinted().call() == 1050
    assert new.functions.balanceOf(acc[3]).call() == 0

    env = gen.rep_env(entries)
    assert env["MEMBERS"] == ",".join(a for a, _ in entries)
    assert env["AMOUNTS"] == ",".join(str(v) for _, v in entries)


def test_empty_snapshot_is_zero_root():
    root, claims = gen.build_tree([])
    assert root == b"\x00" * 32 and claims == {}
