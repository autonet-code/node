"""Governor-only anchoring + on-chain conservation (Decision 2026-10-16).

  - ``Substrate.anchor`` reverts for any sender but the governor; the
    governor's anchor lands (direct and through the real EpochAnchorer).
  - The hash chain still binds: a stale prevEpochRoot / prevAnchorHash
    reverts even from the governor, and an epoch anchors once.
  - floor_mint_to_pool: integer shares never exceed the pool, and the
    merkle leaves built from them sum to <= the on-chain pool exactly.
  - An honest anchor at the seeded pool lets every leaf claim; the claims
    sum to <= mintPool.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

eth_tester = pytest.importorskip("eth_tester")

from eth_account import Account
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from nodes.common.authoritative_submitter import epoch_id_hash
from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.federated_reconcile import floor_mint_to_pool
from nodes.common.mint_merkle import mint_leaves, mint_merkle_proof
from tests.substrate_chain_helpers import (
    ZERO,
    anchor_direct,
    deploy_substrate,
    register_bound,
    seed_pool,
)

ARTIFACT = (Path(__file__).resolve().parent.parent / "artifacts" / "contracts"
            / "core" / "Substrate.sol" / "Substrate.json")


# ---------------------------------------------------------------------------
# Pure: integer flooring to the pool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_floor_mint_to_pool_never_exceeds_pool(seed):
    rng = random.Random(seed)
    pool_raw = rng.randint(1, 10 ** 13)
    pool = pool_raw / 1_000_000
    n = rng.randint(1, 40)
    weights = [rng.random() for _ in range(n)]
    total = sum(weights)
    addrs = [Account.create().address for _ in range(n)]
    mint = {a: pool * w / total for a, w in zip(addrs, weights)}
    floored = floor_mint_to_pool(mint, pool)
    leaves = mint_leaves(floored)
    s = sum(amt for _, amt in leaves)
    assert s <= pool_raw
    assert pool_raw - s <= n


def test_floor_mint_to_pool_trims_float_overshoot():
    pool = 3.0
    mint = {"0x" + "1" * 40: 1.5000004, "0x" + "2" * 40: 1.5000004}
    floored = floor_mint_to_pool(mint, pool)
    assert sum(amt for _, amt in mint_leaves(floored)) == 3_000_000


# ---------------------------------------------------------------------------
# On-chain
# ---------------------------------------------------------------------------


@pytest.fixture
def chain():
    if not ARTIFACT.exists():
        pytest.skip(f"missing artifact: {ARTIFACT}")
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    w3 = Web3(EthereumTesterProvider())
    deployer = w3.eth.accounts[0]
    addr = deploy_substrate(w3, data["abi"], data["bytecode"], deployer)
    contract = w3.eth.contract(address=addr, abi=data["abi"])
    agents = w3.eth.accounts[1:4]
    for i, a in enumerate(agents):
        assert register_bound(w3, contract, a, f"peer-{i}".encode()).status == 1
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=addr),
        web3=w3, contract_abi=data["abi"],
    )
    return {"w3": w3, "contract": contract, "deployer": deployer,
            "agents": list(agents), "anchorer": anchorer, "data": data}


def _close(mint, epoch_id):
    payload = {
        "schema": 3, "epoch_root": "22" * 32, "agent_mint": mint,
        "agent_novelty": {}, "total_mint": sum(mint.values()),
        "total_novelty": 0.0, "output_decimals": 10, "gate_applied": False,
        "n_batches": 1, "n_events": len(mint),
    }
    return {"authoritative_payload": payload, "agent_mint": mint,
            "epoch_id": epoch_id}


def _reverts(fn):
    try:
        receipt = fn()
    except Exception:
        return True
    return getattr(receipt, "status", 1) == 0


def test_governor_is_the_deployer(chain):
    assert chain["contract"].functions.governor().call() == chain["deployer"]


def test_zero_governor_rejected_at_deploy(chain):
    w3, data = chain["w3"], chain["data"]
    assert _reverts(lambda: deploy_substrate(
        w3, data["abi"], data["bytecode"], chain["deployer"], governor=ZERO))


def test_non_governor_anchor_reverts_governor_succeeds(chain):
    w3, c = chain["w3"], chain["contract"]
    for outsider in chain["agents"]:
        assert _reverts(lambda: anchor_direct(
            w3, c, "e1", b"\x01" * 32, "cid-1", b"\x02" * 32, b"\x00" * 32,
            sender=outsider))
    assert c.functions.anchorCount().call() == 0
    r = anchor_direct(w3, c, "e1", b"\x01" * 32, "cid-1", b"\x02" * 32,
                      b"\x00" * 32)
    assert r.status == 1
    assert c.functions.isAnchored("e1").call()
    assert c.functions.anchorCount().call() == 1


def test_epoch_anchorer_outsider_rejected_governor_accepted(chain):
    anchorer, c = chain["anchorer"], chain["contract"]
    attacker = chain["agents"][0]
    evil = anchorer.anchor_close_result(
        _close({attacker: 10 ** 9}, "evil"), from_address=attacker)
    assert not evil.success
    assert c.functions.anchorCount().call() == 0
    ok = anchorer.anchor_close_result(
        _close({attacker: 0.0}, "honest"), from_address=chain["deployer"])
    assert ok.success, ok.error
    assert c.functions.isAnchored("honest").call()


def test_hash_chain_still_binds_the_governor(chain):
    w3, c, gov = chain["w3"], chain["contract"], chain["deployer"]
    assert anchor_direct(w3, c, "e1", b"\x01" * 32, "cid-1", b"\x02" * 32,
                         b"\x00" * 32).status == 1
    head_root = bytes(c.functions.latestEpochRoot().call())
    head_hash = bytes(c.functions.latestAnchorHash().call())

    def submit(epoch_id, prev_root, prev_hash):
        tx = c.functions.anchor(
            epoch_id, b"\x03" * 32, prev_root, prev_hash, "cid-2",
            b"\x04" * 32, b"\x00" * 32,
            int(w3.eth.get_block("latest")["timestamp"])).transact(
                {"from": gov, "gas": 3_000_000})
        return w3.eth.wait_for_transaction_receipt(tx)

    # Stale prevEpochRoot (genesis zero) and stale prevAnchorHash revert.
    assert _reverts(lambda: submit("e2", b"\x00" * 32, head_hash))
    assert _reverts(lambda: submit("e2", head_root, b"\x00" * 32))
    # Re-anchoring an epoch id reverts.
    assert _reverts(lambda: submit("e1", head_root, head_hash))
    assert c.functions.anchorCount().call() == 1
    assert submit("e2", head_root, head_hash).status == 1
    assert c.functions.anchorCount().call() == 2


def test_pool_floored_close_claims_fully_and_conserves(chain):
    w3, c, anchorer = chain["w3"], chain["contract"], chain["anchorer"]
    pool_raw = seed_pool(w3, c, gross=10 ** 9 + 7)
    pool = pool_raw / 1_000_000
    agents = chain["agents"]
    w = [0.37, 1.11, 2.9]
    mint = {a: pool * x / sum(w) for a, x in zip(agents, w)}
    mint = floor_mint_to_pool(mint, pool)
    r = anchorer.anchor_close_result(
        _close(mint, "e-cons"), from_address=chain["deployer"])
    assert r.success, r.error
    anchor = c.functions.getAnchorByEpochId("e-cons").call()
    assert anchor[11] == pool_raw          # mintPool
    eid = epoch_id_hash("e-cons")
    total = 0
    for addr, amount in mint_leaves(mint):
        proof = mint_merkle_proof(mint, addr)
        # claimFor is permissionless: the deployer (an operator key) sends
        # every claim; the ATN lands on the leaf's claimant.
        before = c.functions.balanceOf(addr).call()
        tx = c.functions.claimFor(addr, amount, eid, proof).transact(
            {"from": chain["deployer"], "gas": 1_500_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        assert c.functions.balanceOf(addr).call() == before + amount
        total += amount
    assert total <= pool_raw
    assert c.functions.getAnchorByEpochId("e-cons").call()[12] == total
