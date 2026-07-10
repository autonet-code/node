"""Merkle mint proofs: python tree ↔ Substrate.sol verification.

The anchor commits agentMintRoot; recordTrainingForEpoch only accepts
(msg.sender, amount) pairs that prove into it. Money only (Decision
2026-07-10): the leaf commits the ATN amount alone — REP is claimed
DAO-side (RepToken) on ratified ATN earnings, not minted on this path.
These tests pin:

  - python tree mechanics (roots, proofs, exclusions, determinism),
  - the contract accepting exactly the ratified amounts,
  - the contract rejecting inflated ATN amounts, foreign proofs, and
    claims against rootless (empty-root) anchors,
  - the full submitter path building proofs transparently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

eth_tester = pytest.importorskip("eth_tester")

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from nodes.common.authoritative_submitter import (
    AuthoritativeChainSubmitter,
    AuthoritativeSubmitterConfig,
    epoch_id_hash,
)
from nodes.common.blob_resolver import InMemoryBlobResolver
from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.mint_merkle import (
    EMPTY_ROOT,
    mint_leaves,
    mint_merkle_proof,
    mint_merkle_root,
    scale_mint,
    verify_mint_proof,
)

ARTIFACT = Path("C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json")

ADDRS = [
    "0x" + f"{i:040x}" for i in range(1, 8)
]


# ---------------------------------------------------------------------------
# Python tree mechanics
# ---------------------------------------------------------------------------


def test_round_trip_all_tree_sizes():
    for n in range(1, 8):
        mint = {ADDRS[i]: float(i + 1) * 1.5 for i in range(n)}
        root = mint_merkle_root(mint)
        for addr, raw in mint.items():
            proof = mint_merkle_proof(mint, addr)
            assert proof is not None, (n, addr)
            assert verify_mint_proof(root, addr, scale_mint(raw), proof), (n, addr)


def test_single_leaf_root_is_leaf_with_empty_proof():
    mint = {ADDRS[0]: 2.5}
    root = mint_merkle_root(mint)
    proof = mint_merkle_proof(mint, ADDRS[0])
    assert proof == []
    assert verify_mint_proof(root, ADDRS[0], scale_mint(2.5), [])


def test_wrong_amount_fails_verification():
    mint = {ADDRS[0]: 1.0, ADDRS[1]: 2.0}
    root = mint_merkle_root(mint)
    proof = mint_merkle_proof(mint, ADDRS[0])
    assert not verify_mint_proof(root, ADDRS[0], scale_mint(1.0) + 1, proof)


def test_non_address_and_zero_keys_excluded():
    mint = {"alice": 5.0, ADDRS[0]: 0.0, ADDRS[1]: 3.0}
    leaves = mint_leaves(mint)
    assert len(leaves) == 1
    assert leaves[0][0].lower() == ADDRS[1].lower()
    assert mint_merkle_proof(mint, "alice") is None
    assert mint_merkle_proof(mint, ADDRS[0]) is None   # zero share


def test_empty_map_yields_empty_root():
    assert mint_merkle_root({}) == EMPTY_ROOT
    assert mint_merkle_root({"not-an-address": 9.0}) == EMPTY_ROOT


def test_root_independent_of_key_order():
    mint_a = {ADDRS[0]: 1.0, ADDRS[1]: 2.0, ADDRS[2]: 3.0}
    mint_b = dict(reversed(list(mint_a.items())))
    assert mint_merkle_root(mint_a) == mint_merkle_root(mint_b)


def test_two_field_leaf_shape():
    """Money-only (Decision 2026-07-10): a leaf entry is (address, amount)
    — a 2-tuple, no repAmount third field."""
    mint = {ADDRS[0]: 4.0}
    leaves = mint_leaves(mint)
    assert len(leaves[0]) == 2
    assert leaves[0][1] == scale_mint(4.0)


# ---------------------------------------------------------------------------
# Contract enforcement
# ---------------------------------------------------------------------------


@pytest.fixture
def chain():
    if not ARTIFACT.exists():
        pytest.skip(f"missing artifact: {ARTIFACT}")
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    w3 = Web3(EthereumTesterProvider())
    deployer = w3.eth.accounts[0]
    zero = "0x0000000000000000000000000000000000000000"
    contract_f = w3.eth.contract(abi=data["abi"], bytecode=data["bytecode"])
    # Substrate(treasury, vaultMinter, governor). governor=zero => fee lever
    # frozen; irrelevant to the mint-proof path.
    tx = contract_f.constructor(deployer, zero, zero).transact(
        {"from": deployer, "gas": 8_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    contract = w3.eth.contract(address=receipt.contractAddress, abi=data["abi"])
    agent_addrs = w3.eth.accounts[1:4]
    for i, a in enumerate(agent_addrs):
        tx = contract.functions.registerAgent(
            Web3.keccak(text=f"lineage-{i}"), f"peer-{i}".encode(),
        ).transact({"from": a, "gas": 500_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
    return {
        "w3": w3, "abi": data["abi"], "addr": contract.address,
        "contract": contract, "deployer": deployer,
        "agent_addrs": list(agent_addrs),
    }


def _anchor_with_mint(chain, mint: Dict[str, float], epoch_id: str):
    """Anchor a hand-built close result carrying this mint map.

    Money-only (schema 3): no agent_rep."""
    payload = {
        "schema": 3,
        "epoch_root": "11" * 32,
        "agent_mint": mint,
        "agent_novelty": {},
        "total_mint": sum(mint.values()),
        "total_novelty": 0.0,
        "output_decimals": 10,
        "gate_applied": True,
        "n_batches": 1,
        "n_events": len(mint),
    }
    result = {
        "authoritative_payload": payload,
        "agent_mint": mint,
        "epoch_id": epoch_id,
    }
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=chain["w3"], contract_abi=chain["abi"],
    )
    s = anchorer.anchor_close_result(result, from_address=chain["deployer"])
    assert s.success, s.error
    return s


def _record(chain, agent, amount, eid_hash, proof):
    # Money only (Decision 2026-07-10): recordTrainingForEpoch takes
    # (amount, epochIdHash, proof); the leaf commits (agent, amount).
    fn = chain["contract"].functions.recordTrainingForEpoch(
        amount, eid_hash, proof,
    )
    tx = fn.transact({"from": agent, "gas": 1_500_000})
    receipt = chain["w3"].eth.wait_for_transaction_receipt(tx)
    return receipt.status == 1


def test_contract_accepts_ratified_amounts(chain):
    addrs = chain["agent_addrs"]
    mint = {addrs[0]: 1.5, addrs[1]: 2.25, addrs[2]: 0.75}
    s = _anchor_with_mint(chain, mint, "e_merkle_ok")
    assert s.agent_mint_root_hex != "00" * 32
    eid = epoch_id_hash("e_merkle_ok")

    for a in addrs:
        proof = mint_merkle_proof(mint, a)
        scaled = scale_mint(mint[a])
        assert _record(chain, a, scaled, eid, proof)
        assert chain["contract"].functions.mintForEpoch(a, eid).call() == scaled


def test_contract_mints_atn_at_amount(chain):
    """recordTrainingForEpoch mints ATN at the proven amount and bumps
    agentMintTotal (cumulative tool-pool earnings). No reputation surface
    exists on the money-only contract."""
    contract = chain["contract"]
    addrs = chain["agent_addrs"]
    mint = {addrs[0]: 4.0, addrs[1]: 1.0}
    s = _anchor_with_mint(chain, mint, "e_atn")
    eid = epoch_id_hash("e_atn")

    a = addrs[0]
    amount = scale_mint(mint[a])
    proof = mint_merkle_proof(mint, a)
    assert _record(chain, a, amount, eid, proof)

    assert contract.functions.balanceOf(a).call() == amount
    assert contract.functions.atnTotalSupply().call() == amount
    assert contract.functions.mintForEpoch(a, eid).call() == amount
    # agentMintTotal = cumulative pool earnings (money).
    assert contract.functions.agentMintTotal(a).call() == amount
    assert contract.functions.networkMintTotal().call() == amount


def test_contract_rejects_inflated_amount(chain):
    addrs = chain["agent_addrs"]
    mint = {addrs[0]: 1.0, addrs[1]: 2.0}
    _anchor_with_mint(chain, mint, "e_merkle_inflate")
    eid = epoch_id_hash("e_merkle_inflate")

    proof = mint_merkle_proof(mint, addrs[0])
    rejected = False
    try:
        ok = _record(chain, addrs[0], scale_mint(1.0) * 10, eid, proof)
        rejected = not ok
    except Exception:
        rejected = True
    assert rejected, "MintProofInvalid was not enforced for an inflated amount"
    assert chain["contract"].functions.mintForEpoch(addrs[0], eid).call() == 0


def test_contract_rejects_foreign_proof(chain):
    addrs = chain["agent_addrs"]
    mint = {addrs[0]: 1.0, addrs[1]: 2.0}
    _anchor_with_mint(chain, mint, "e_merkle_foreign")
    eid = epoch_id_hash("e_merkle_foreign")

    # addrs[1] tries to claim addrs[0]'s amount using addrs[0]'s proof.
    proof = mint_merkle_proof(mint, addrs[0])
    rejected = False
    try:
        ok = _record(chain, addrs[1], scale_mint(1.0), eid, proof)
        rejected = not ok
    except Exception:
        rejected = True
    assert rejected, "proof bound to another agent was accepted"


def test_contract_rejects_rootless_anchor_claims(chain):
    """An anchor whose mint map has no claimable entries carries the
    empty root; the contract refuses ALL nonzero claims against it."""
    addrs = chain["agent_addrs"]
    s = _anchor_with_mint(chain, {"not-an-address": 5.0}, "e_merkle_rootless")
    assert s.agent_mint_root_hex == "00" * 32
    eid = epoch_id_hash("e_merkle_rootless")

    rejected = False
    try:
        ok = _record(chain, addrs[0], 123, eid, [])
        rejected = not ok
    except Exception:
        rejected = True
    assert rejected, "MintRootMissing was not enforced"


def test_submitter_builds_proof_end_to_end(chain):
    """Full path: anchor → blob → submitter decodes, proves, claims."""
    addrs = chain["agent_addrs"]
    mint = {addrs[0]: 3.5, addrs[1]: 1.25}
    s = _anchor_with_mint(chain, mint, "e_merkle_sub")
    resolver = InMemoryBlobResolver()
    resolver.put(s.agent_mint_blob)

    submitter = AuthoritativeChainSubmitter(
        agent_id=addrs[0], agent_address=addrs[0],
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"], substrate_abi=chain["abi"],
    )
    sub = submitter.submit_for_epoch("e_merkle_sub")
    assert sub.success, sub.error
    assert sub.contribution_scaled == scale_mint(3.5)
    eid = epoch_id_hash("e_merkle_sub")
    assert chain["contract"].functions.mintForEpoch(
        addrs[0], eid).call() == scale_mint(3.5)
