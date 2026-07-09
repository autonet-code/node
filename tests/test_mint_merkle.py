"""Merkle mint proofs: python tree ↔ Substrate.sol verification.

The anchor commits agentMintRoot; recordTrainingForEpoch only accepts
(msg.sender, amount, repAmount) triples that prove into it (v4.1
gradient trust: the leaf commits BOTH the ATN and the reputation
amounts, so voice is federation-ratified too). These tests pin:

  - python tree mechanics (roots, proofs, exclusions, determinism,
    decoupled-rep leaves, lockstep default equivalence),
  - the contract accepting exactly the ratified amounts,
  - the contract minting the two ledgers at decoupled amounts,
  - the contract rejecting inflated ATN amounts, inflated/deflated
    repAmounts (the ring-author voice-bootstrap attack), foreign
    proofs, and claims against rootless (legacy-shaped) anchors,
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


def test_decoupled_rep_round_trip():
    """v4.1: leaves commit (agent, amount, repAmount). A decoupled rep
    map produces a DIFFERENT root than lockstep; proofs verify only with
    the exact (amount, rep_amount) pair the root ratified; a missing key
    in agent_rep means zero reputation (legit: all-zero-rep callers)."""
    mint = {ADDRS[0]: 4.0, ADDRS[1]: 2.0}
    rep = {ADDRS[0]: 1.0}                    # ADDRS[1] absent -> 0 rep
    root = mint_merkle_root(mint, agent_rep=rep)
    # Decoupled root differs from the lockstep root over the same mint.
    assert root != mint_merkle_root(mint)

    p0 = mint_merkle_proof(mint, ADDRS[0], agent_rep=rep)
    assert verify_mint_proof(root, ADDRS[0], scale_mint(4.0), p0,
                             rep_amount=scale_mint(1.0))
    # Lying about rep (claiming lockstep rep == amount) fails.
    assert not verify_mint_proof(root, ADDRS[0], scale_mint(4.0), p0)
    assert not verify_mint_proof(root, ADDRS[0], scale_mint(4.0), p0,
                                 rep_amount=scale_mint(1.0) + 1)

    # Zero-rep leaf (positive ATN, no voice) proves with rep_amount=0.
    p1 = mint_merkle_proof(mint, ADDRS[1], agent_rep=rep)
    assert verify_mint_proof(root, ADDRS[1], scale_mint(2.0), p1,
                             rep_amount=0)


def test_lockstep_default_equals_explicit_lockstep_rep():
    """agent_rep=None (lockstep) is byte-identical to passing an
    explicit rep map equal to the mint map — the back-compat contract
    the submitter/anchorer rely on until the close threads real rep."""
    mint = {ADDRS[0]: 1.5, ADDRS[1]: 2.25, ADDRS[2]: 0.75}
    assert mint_merkle_root(mint) == mint_merkle_root(mint, agent_rep=mint)
    for a in mint:
        assert (mint_merkle_proof(mint, a)
                == mint_merkle_proof(mint, a, agent_rep=mint))


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
    contract_f = w3.eth.contract(abi=data["abi"], bytecode=data["bytecode"])
    tx = contract_f.constructor(deployer, "0x0000000000000000000000000000000000000000").transact({"from": deployer, "gas": 8_000_000})
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


def _anchor_with_mint(chain, mint: Dict[str, float], epoch_id: str,
                      rep: Dict[str, float] | None = None):
    """Anchor a hand-built close result carrying this mint map.

    ``rep`` (v4.1, schema 2) is the close's decoupled agent_rep map;
    None = legacy/lockstep close (rep == mint per leaf)."""
    payload = {
        "schema": 2,
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
    if rep is not None:
        payload["agent_rep"] = rep
        result["agent_rep"] = rep
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=chain["w3"], contract_abi=chain["abi"],
    )
    s = anchorer.anchor_close_result(result, from_address=chain["deployer"])
    assert s.success, s.error
    return s


def _anchor_with_root(chain, root: bytes, epoch_id: str):
    """Anchor an epoch carrying an explicit agentMintRoot — used for
    DECOUPLED-rep roots, which EpochAnchorer (lockstep default) doesn't
    build yet. Links prev root/hash off the live contract state."""
    contract = chain["contract"]
    w3 = chain["w3"]
    prev_root = bytes(contract.functions.latestEpochRoot().call())
    prev_anchor = bytes(contract.functions.latestAnchorHash().call())
    tx = contract.functions.submitAnchor(
        epoch_id,
        Web3.keccak(text=f"root-{epoch_id}"),      # epochRoot (opaque here)
        prev_root,
        prev_anchor,
        f"cid-{epoch_id}",                          # agentMintCid (opaque)
        Web3.keccak(text=f"payload-{epoch_id}"),
        root,
    ).transact({"from": chain["deployer"], "gas": 1_500_000})
    assert w3.eth.wait_for_transaction_receipt(tx).status == 1


def _record(chain, agent, amount, eid_hash, proof, rep_amount=None):
    # v4.1 gradient-trust: recordTrainingForEpoch takes
    # (amount, repAmount, epochIdHash, proof) and the merkle leaf commits
    # (agent, amount, repAmount). ``rep_amount`` defaults to ``amount``
    # (legacy lockstep), matching mint_merkle's default lockstep leaves.
    if rep_amount is None:
        rep_amount = amount
    fn = chain["contract"].functions.recordTrainingForEpoch(
        amount, rep_amount, eid_hash, proof,
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


def test_rep_decoupled_from_atn(chain):
    """v4.1 gradient trust: recordTrainingForEpoch mints ATN at ``amount``
    but reputation at a DISTINCT, smaller ``repAmount`` — and BOTH are
    ratified by the anchor: the root is built over 3-field
    (agent, amount, repAmount) leaves. Assert the money ledger
    (balanceOf/atnTotalSupply) tracks amount and the voice ledger
    (agentReputation + its checkpoint) tracks repAmount."""
    w3 = chain["w3"]
    contract = chain["contract"]
    addrs = chain["agent_addrs"]
    # The federation ratified: addrs[0] earns 4.0 ATN but only 1.0 rep
    # (the rest of its usage came from zero-rep callers).
    mint = {addrs[0]: 4.0, addrs[1]: 1.0}
    rep = {addrs[0]: 1.0, addrs[1]: 1.0}
    root = mint_merkle_root(mint, agent_rep=rep)
    _anchor_with_root(chain, root, "e_decouple")
    eid = epoch_id_hash("e_decouple")

    a = addrs[0]
    amount = scale_mint(mint[a])            # ATN (money) mint
    rep_amount = scale_mint(rep[a])          # voice mint: strictly < amount
    assert 0 < rep_amount < amount

    proof = mint_merkle_proof(mint, a, agent_rep=rep)
    assert _record(chain, a, amount, eid, proof, rep_amount=rep_amount)
    mint_block = w3.eth.block_number

    # Money ledger tracks the full ATN amount.
    assert contract.functions.balanceOf(a).call() == amount
    assert contract.functions.atnTotalSupply().call() == amount
    assert contract.functions.balanceOfAt(a, mint_block).call() == amount
    # mintForEpoch records the ATN amount (what the proof committed).
    assert contract.functions.mintForEpoch(a, eid).call() == amount

    # Voice ledger tracks only repAmount — decoupled and smaller.
    assert contract.functions.agentReputation(a).call() == rep_amount
    assert contract.functions.agentMintTotal(a).call() == rep_amount
    assert contract.functions.networkMintTotal().call() == rep_amount
    # The reputation CHECKPOINT reflects repAmount, not amount.
    assert contract.functions.reputationOfAt(a, mint_block).call() == rep_amount
    assert (contract.functions.reputationTotalSupplyAt(mint_block).call()
            == rep_amount)


def test_lying_about_rep_amount_fails_proof(chain):
    """THE v4.1 ratification property: repAmount is committed in the
    merkle leaf, so a claimant cannot self-report more voice than the
    federation attributed. The ring-author attack — claiming
    repAmount == amount when the close ratified less — must fail proof
    verification (MintProofInvalid), not just the <= ceiling."""
    contract = chain["contract"]
    addrs = chain["agent_addrs"]
    mint = {addrs[0]: 4.0, addrs[1]: 1.0}
    rep = {addrs[0]: 1.0, addrs[1]: 1.0}     # ratified: 1.0 rep, not 4.0
    root = mint_merkle_root(mint, agent_rep=rep)
    _anchor_with_root(chain, root, "e_rep_lie")
    eid = epoch_id_hash("e_rep_lie")

    a = addrs[0]
    amount = scale_mint(mint[a])
    ratified_rep = scale_mint(rep[a])
    proof = mint_merkle_proof(mint, a, agent_rep=rep)

    def _rejected(rep_claim):
        try:
            return not _record(chain, a, amount, eid, proof,
                               rep_amount=rep_claim)
        except Exception:
            return True

    # Claiming lockstep voice (repAmount == amount) with the honest
    # proof: the 3-field leaf doesn't match the ratified one.
    assert _rejected(amount), "inflated repAmount == amount was accepted"
    # Any other inflation above the ratified rep fails too.
    assert _rejected(ratified_rep + 1), "repAmount + 1 was accepted"
    # ...and so does deflation (the leaf commits the exact pair).
    assert _rejected(ratified_rep - 1), "repAmount - 1 was accepted"
    # Nothing minted on either ledger by the failed attempts.
    assert contract.functions.mintForEpoch(a, eid).call() == 0
    assert contract.functions.balanceOf(a).call() == 0
    assert contract.functions.agentReputation(a).call() == 0

    # The exact ratified pair still claims fine.
    assert _record(chain, a, amount, eid, proof, rep_amount=ratified_rep)
    assert contract.functions.balanceOf(a).call() == amount
    assert contract.functions.agentReputation(a).call() == ratified_rep


def test_rep_exceeding_amount_reverts(chain):
    """repAmount > amount must revert (RepExceedsAmount) — reputation can
    never exceed the ATN earned in the same event. No ledger moves."""
    contract = chain["contract"]
    addrs = chain["agent_addrs"]
    mint = {addrs[0]: 2.0, addrs[1]: 1.0}
    _anchor_with_mint(chain, mint, "e_rep_over")
    eid = epoch_id_hash("e_rep_over")

    a = addrs[0]
    amount = scale_mint(mint[a])
    proof = mint_merkle_proof(mint, a)

    rejected = False
    try:
        ok = _record(chain, a, amount, eid, proof, rep_amount=amount + 1)
        rejected = not ok
    except Exception:
        rejected = True
    assert rejected, "RepExceedsAmount was not enforced for repAmount > amount"
    # Nothing minted on either ledger.
    assert contract.functions.mintForEpoch(a, eid).call() == 0
    assert contract.functions.balanceOf(a).call() == 0
    assert contract.functions.agentReputation(a).call() == 0

    # Sanity: repAmount == amount (the equality boundary) is accepted.
    assert _record(chain, a, amount, eid, proof, rep_amount=amount)
    assert contract.functions.agentReputation(a).call() == amount
    assert contract.functions.balanceOf(a).call() == amount


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
    # Legacy/lockstep close (no agent_rep): rep mints in lockstep.
    assert sub.rep_contribution_scaled == scale_mint(3.5)
    eid = epoch_id_hash("e_merkle_sub")
    assert chain["contract"].functions.mintForEpoch(
        addrs[0], eid).call() == scale_mint(3.5)


def test_submitter_decoupled_rep_end_to_end(chain):
    """v4.1 full glue: a schema-2 close result with a decoupled
    ``agent_rep`` map → EpochAnchorer builds root AND blob from the
    SAME map → the per-agent submitter decodes the blob, rebuilds the
    proof, pre-flight-verifies it against the on-chain root, and
    records with rep < amount. A lockstep (inflated-rep) claim against
    the same anchor fails proof verification; the ratified zero-rep
    pair still claims."""
    addrs = chain["agent_addrs"]
    contract = chain["contract"]
    mint = {addrs[0]: 3.5, addrs[1]: 2.0}
    rep = {addrs[0]: 0.5}                # addrs[1] absent → ZERO voice
    s = _anchor_with_mint(chain, mint, "e_merkle_rep_e2e", rep=rep)
    resolver = InMemoryBlobResolver()
    resolver.put(s.agent_mint_blob)

    # The anchored root is the DECOUPLED root, not the lockstep one.
    assert s.agent_mint_root_hex == mint_merkle_root(
        mint, agent_rep=rep).hex()
    assert s.agent_mint_root_hex != mint_merkle_root(mint).hex()

    # addrs[0] submits through the production submitter path.
    submitter = AuthoritativeChainSubmitter(
        agent_id=addrs[0], agent_address=addrs[0],
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"], substrate_abi=chain["abi"],
    )
    sub = submitter.submit_for_epoch("e_merkle_rep_e2e")
    assert sub.success, sub.error
    amount = scale_mint(3.5)
    rep_amount = scale_mint(0.5)
    assert sub.contribution_scaled == amount
    assert sub.rep_contribution_scaled == rep_amount

    # Money ledger tracks amount; voice ledger tracks the smaller rep.
    assert contract.functions.balanceOf(addrs[0]).call() == amount
    assert contract.functions.agentReputation(addrs[0]).call() == rep_amount

    # Mismatched rep claim: addrs[1] tries lockstep voice (rep == its
    # amount) instead of the ratified ZERO — unprovable leaf, reverts.
    eid = epoch_id_hash("e_merkle_rep_e2e")
    proof1 = mint_merkle_proof(mint, addrs[1], agent_rep=rep)
    amt1 = scale_mint(2.0)
    rejected = False
    try:
        ok = _record(chain, addrs[1], amt1, eid, proof1, rep_amount=amt1)
        rejected = not ok
    except Exception:
        rejected = True
    assert rejected, "lockstep rep claim against a decoupled anchor accepted"
    assert contract.functions.agentReputation(addrs[1]).call() == 0
    assert contract.functions.balanceOf(addrs[1]).call() == 0

    # The ratified pair (zero rep) claims fine: money without voice.
    assert _record(chain, addrs[1], amt1, eid, proof1, rep_amount=0)
    assert contract.functions.balanceOf(addrs[1]).call() == amt1
    assert contract.functions.agentReputation(addrs[1]).call() == 0
