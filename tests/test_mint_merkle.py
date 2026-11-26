"""Claim proofs: the python tree and Substrate's verification.

The anchor commits a claimRoot; ``Substrate.claim`` only accepts an
(earner, tokens, amounts, seed) leaf that proves into it, whoever sends the
transaction (Decision 2026-10-27: the owner wallet earns, the daemon
operates). These tests pin:

  - python tree mechanics (roots, proofs, exclusions, determinism),
  - the leaf against an independent eth_abi + keccak computation and
    against ``Substrate.claimLeaf`` on the compiled contract,
  - the contract accepting exactly the anchored amounts and rejecting
    inflated amounts, foreign proofs and claims against an empty root,
  - the full submitter path: claim, then a withdrawal per token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eth_abi import encode as abi_encode
from eth_utils import keccak, to_checksum_address

from nodes.common.mint_merkle import (
    CLAIM_TAG,
    EMPTY_ROOT,
    claim_leaf,
    claim_leaves,
    claim_merkle_proof,
    claim_merkle_root,
    concluded_leaf,
    concluded_proof,
    verify_claim_proof,
    verify_concluded_proof,
)

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts/contracts/core/Substrate.sol/Substrate.json"

ADDRS = ["0x" + f"{i:040x}" for i in range(1, 8)]
T6 = "0x" + "a6" * 20      # stands for a 6-decimal token
T18 = "0x" + "1b" * 20     # stands for an 18-decimal token


def _claims(n: int):
    return {ADDRS[i]: {"tokens": {T6: (i + 1) * 1_500_000, T18: (i + 1) * 10 ** 18},
                       "value": (i + 1) * 2_500_000} for i in range(n)}


def _fields(claims, earner):
    """(tokens, amounts, seed): the leaf's fields after the earner."""
    tokens = sorted(t.lower() for t in claims[earner]["tokens"])
    return (tokens, [claims[earner]["tokens"][t] for t in tokens],
            int(claims[earner].get("seed") or 0))


# ---------------------------------------------------------------------------
# Python tree mechanics
# ---------------------------------------------------------------------------


def test_round_trip_all_tree_sizes():
    for n in range(1, 8):
        claims = _claims(n)
        root = claim_merkle_root(claims)
        for addr in claims:
            proof = claim_merkle_proof(claims, addr)
            assert proof is not None, (n, addr)
            assert verify_claim_proof(root, addr, *_fields(claims, addr), proof), (n, addr)


def test_single_leaf_root_is_leaf_with_empty_proof():
    claims = _claims(1)
    root = claim_merkle_root(claims)
    assert claim_merkle_proof(claims, ADDRS[0]) == []
    assert root == claim_leaf(ADDRS[0], *_fields(claims, ADDRS[0]))


def test_wrong_amount_fails_verification():
    claims = _claims(2)
    root = claim_merkle_root(claims)
    proof = claim_merkle_proof(claims, ADDRS[0])
    tokens, amounts, seed = _fields(claims, ADDRS[0])
    assert verify_claim_proof(root, ADDRS[0], tokens, amounts, seed, proof)
    assert not verify_claim_proof(root, ADDRS[0], tokens, [amounts[0] + 1, amounts[1]], seed, proof)
    assert not verify_claim_proof(root, ADDRS[0], tokens[:1], amounts[:1], seed, proof)
    assert not verify_claim_proof(root, ADDRS[0], tokens, amounts, seed + 1, proof)


def test_non_address_and_zero_amounts_excluded():
    claims = {"alice": {"tokens": {T6: 5}, "value": 5},
              ADDRS[0]: {"tokens": {T6: 0}, "value": 0},
              ADDRS[1]: {"tokens": {T6: 3, T18: 0}, "value": 3}}
    leaves = claim_leaves(claims)
    assert len(leaves) == 1
    earner, tokens, amounts, seed = leaves[0]
    assert earner.lower() == ADDRS[1] and tokens == [to_checksum_address(T6)] and amounts == [3]
    assert seed == 0
    assert claim_merkle_proof(claims, "alice") is None
    assert claim_merkle_proof(claims, ADDRS[0]) is None


def test_a_seed_alone_makes_a_leaf_and_is_in_it():
    """The REP floor: a household with no tokens but a seed has a leaf, and
    the seed is committed (a different seed is a different leaf)."""
    claims = {ADDRS[0]: {"tokens": {}, "value": 0, "seed": 40},
              ADDRS[1]: {"tokens": {T6: 3}, "value": 3, "seed": 0}}
    leaves = claim_leaves(claims)
    assert [(l[0].lower(), l[1], l[2], l[3]) for l in leaves] == [
        (ADDRS[0], [], [], 40), (ADDRS[1], [to_checksum_address(T6)], [3], 0)]
    root = claim_merkle_root(claims)
    proof = claim_merkle_proof(claims, ADDRS[0])
    assert verify_claim_proof(root, ADDRS[0], [], [], 40, proof)
    assert not verify_claim_proof(root, ADDRS[0], [], [], 41, proof)
    bumped = {**claims, ADDRS[0]: {**claims[ADDRS[0]], "seed": 41}}
    assert claim_merkle_root(bumped) != root


def test_empty_claims_yield_empty_root():
    assert claim_merkle_root({}) == EMPTY_ROOT
    assert claim_merkle_root({"not-an-address": {"tokens": {T6: 9}, "value": 9}}) == EMPTY_ROOT


def test_root_independent_of_key_order_and_case():
    a = _claims(3)
    b = {k.upper().replace("0X", "0x"): {"tokens": dict(reversed(list(v["tokens"].items()))),
                                         "value": v["value"]}
         for k, v in reversed(list(a.items()))}
    assert claim_merkle_root(a) == claim_merkle_root(b)


def test_value_is_not_in_the_leaf():
    """Substrate computes REP on chain; the close's value is informational."""
    a = _claims(2)
    b = {k: {**v, "value": v["value"] + 7} for k, v in a.items()}
    assert claim_merkle_root(a) == claim_merkle_root(b)


def test_leaf_matches_an_independent_abi_encoding_two_tokens():
    """keccak256(bytes.concat(keccak256(abi.encode(CLAIM_TAG, earner,
    tokens, amounts, seed)))), tokens ascending."""
    earner = "0x" + "c0" * 20
    claims = {earner: {"tokens": {T6: 1_234_567, T18: 5 * 10 ** 17}, "value": 1_734_567,
                       "seed": 265_433}}
    tokens = sorted([T6, T18])                  # lowercase order = numeric order
    amounts = [claims[earner]["tokens"][t] for t in tokens]
    assert tokens == [T18, T6]                  # 0x1b.. < 0xa6..
    expected = keccak(keccak(abi_encode(
        ["bytes32", "address", "address[]", "uint256[]", "uint256"],
        [keccak(text="autonet.claim"), to_checksum_address(earner),
         [to_checksum_address(t) for t in tokens], amounts, 265_433])))
    assert CLAIM_TAG == keccak(text="autonet.claim")
    assert claim_merkle_root(claims) == expected
    assert claim_leaf(earner, tokens, amounts, 265_433) == expected


def test_concluded_leaves_share_the_tree():
    """Decision 2026-11-09: each concluding tool's leaf sits after the claim
    leaves; both kinds prove against the same root."""
    claims = _claims(2)
    concluded = {"bb" * 32: 6000, "0x" + "aa" * 32: 0}
    root = claim_merkle_root(claims, concluded=concluded)
    assert root != claim_merkle_root(claims)
    for a in ADDRS[:2]:
        proof = claim_merkle_proof(claims, a, concluded=concluded)
        assert verify_claim_proof(root, a, *_fields(claims, a), proof)
    for d, bps in (("aa" * 32, 0), ("bb" * 32, 6000)):
        proof = concluded_proof(claims, concluded, d)
        assert verify_concluded_proof(root, d, bps, proof)
        assert not verify_concluded_proof(root, d, bps + 1, proof)
    assert concluded_proof(claims, concluded, "cc" * 32) is None
    only = claim_merkle_root({}, concluded={"bb" * 32: 1})
    assert only == concluded_leaf("bb" * 32, 1)


# ---------------------------------------------------------------------------
# The compiled contract
# ---------------------------------------------------------------------------


@pytest.fixture
def chain():
    pytest.importorskip("eth_tester")
    if not ARTIFACT.exists():
        pytest.skip(f"missing artifact: {ARTIFACT}")
    from web3 import Web3
    from web3.providers.eth_tester import EthereumTesterProvider

    from tests.substrate_chain_helpers import (
        deploy_substrate, index_tokens, register_bound, seed_pool)
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    w3 = Web3(EthereumTesterProvider())
    deployer = w3.eth.accounts[0]
    addr = deploy_substrate(w3, data["abi"], data["bytecode"], deployer)
    contract = w3.eth.contract(address=addr, abi=data["abi"])
    agent_addrs = w3.eth.accounts[1:4]
    for i, a in enumerate(agent_addrs):
        assert register_bound(w3, contract, a, f"peer-{i}".encode()).status == 1
    shares = seed_pool(w3, contract)
    return {"w3": w3, "abi": data["abi"], "addr": contract.address,
            "contract": contract, "deployer": deployer,
            "agent_addrs": list(agent_addrs), "tokens": index_tokens(contract),
            "shares": shares}


def test_leaf_matches_substrate_claim_leaf(chain):
    earner = chain["agent_addrs"][0]
    tokens = sorted(t.address.lower() for t in chain["tokens"])
    amounts = [7, 10 ** 18 + 3]
    for seed in (0, 12_345):
        onchain = chain["contract"].functions.claimLeaf(
            earner, [to_checksum_address(t) for t in tokens], amounts, seed).call()
        assert bytes(onchain) == claim_leaf(earner, tokens, amounts, seed)


def _anchor_claims(chain, claims, epoch_id):
    from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
    payload = {"epoch_root": "11" * 32, "claims": claims, "pools": {}, "parity": {},
               "total_value": sum(c["value"] for c in claims.values()),
               "output_decimals": 10, "n_batches": 1, "n_events": 1}
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=chain["w3"], contract_abi=chain["abi"])
    s = anchorer.anchor_close_result(
        {"authoritative_payload": payload, "claims": claims, "epoch_id": epoch_id},
        from_address=chain["deployer"])
    assert s.success, s.error
    return s


def _two_token_claims(chain, *amounts_per_earner):
    t6, t18 = (t.address.lower() for t in chain["tokens"])
    return {a.lower(): {"tokens": {t6: x6, t18: x18}, "value": x6 + x18 // 10 ** 12}
            for a, (x6, x18) in zip(chain["agent_addrs"], amounts_per_earner)}


def _claim(chain, claims, earner, epoch_id, sender=None, tokens=None, amounts=None):
    from nodes.common.authoritative_submitter import epoch_id_hash
    t, a, seed = _fields(claims, earner.lower())
    proof = claim_merkle_proof(claims, earner)
    tx = chain["contract"].functions.claim(
        earner, epoch_id_hash(epoch_id),
        [to_checksum_address(x) for x in (tokens or t)], amounts or a, seed, proof,
    ).transact({"from": sender or chain["deployer"], "gas": 1_500_000})
    return chain["w3"].eth.wait_for_transaction_receipt(tx).status == 1


def test_contract_accepts_the_anchored_amounts(chain):
    claims = _two_token_claims(chain, (10, 10 ** 9), (20, 3 * 10 ** 9))
    _anchor_claims(chain, claims, "e_ok")
    c = chain["contract"]
    for earner in chain["agent_addrs"][:2]:
        assert _claim(chain, claims, earner, "e_ok")
        # The claim pays the basket in the same transaction.
        for t in chain["tokens"]:
            assert c.functions.owed(earner, t.address).call() == 0
            assert t.functions.balanceOf(earner).call() == \
                claims[earner.lower()]["tokens"][t.address.lower()]


def _reverts(fn):
    try:
        return not fn()
    except Exception:
        return True


def test_contract_rejects_inflated_and_foreign_claims(chain):
    claims = _two_token_claims(chain, (10, 10 ** 9), (20, 3 * 10 ** 9))
    _anchor_claims(chain, claims, "e_bad")
    a0, a1 = chain["agent_addrs"][:2]
    t, amounts, _seed = _fields(claims, a0.lower())
    assert _reverts(lambda: _claim(chain, claims, a0, "e_bad",
                                   amounts=[amounts[0] * 10, amounts[1]]))

    from nodes.common.authoritative_submitter import epoch_id_hash
    proof = claim_merkle_proof(claims, a0)
    assert _reverts(lambda: chain["w3"].eth.wait_for_transaction_receipt(
        chain["contract"].functions.claim(
            a1, epoch_id_hash("e_bad"), [to_checksum_address(x) for x in t], amounts,
            0, proof).transact({"from": a1, "gas": 1_500_000})).status == 1)


def test_contract_rejects_claims_against_an_empty_root(chain):
    s = _anchor_claims(chain, {"not-an-address": {"tokens": {}, "value": 0}}, "e_rootless")
    assert s.agent_mint_root_hex == "00" * 32
    from nodes.common.authoritative_submitter import epoch_id_hash
    a0 = chain["agent_addrs"][0]
    tok = to_checksum_address(chain["tokens"][0].address)
    assert _reverts(lambda: chain["w3"].eth.wait_for_transaction_receipt(
        chain["contract"].functions.claim(a0, epoch_id_hash("e_rootless"), [tok], [1], 0, [])
        .transact({"from": a0, "gas": 1_500_000})).status == 1)


def test_submitter_claims_and_is_paid_each_token(chain):
    """Full path: anchor -> blob -> submitter proves, claims for the
    household, signed by an operator that is not the earner; the claim pays
    each token to the earner, so no withdrawal follows."""
    from nodes.common.authoritative_submitter import (
        AuthoritativeChainSubmitter, AuthoritativeSubmitterConfig)
    from nodes.common.blob_resolver import InMemoryBlobResolver
    claims = _two_token_claims(chain, (35, 2 * 10 ** 9), (12, 10 ** 9))
    s = _anchor_claims(chain, claims, "e_sub")
    resolver = InMemoryBlobResolver()
    resolver.put(s.agent_mint_blob)
    earner = chain["agent_addrs"][0]
    sub = AuthoritativeChainSubmitter(
        claimant_id=earner.lower(), claimant_address=earner, resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        sender_address=chain["deployer"], web3=chain["w3"], substrate_abi=chain["abi"])
    r = sub.submit_for_epoch("e_sub")
    assert r.success, r.error
    assert r.tokens == claims[earner.lower()]["tokens"]
    assert r.tx_hash and not r.withdrawn and not r.withdraw_errors and not r.owed
    for t in chain["tokens"]:
        assert t.functions.balanceOf(earner).call() == r.tokens[t.address.lower()]
        assert chain["contract"].functions.owed(earner, t.address).call() == 0
    # Idempotent: a second pass claims nothing and withdraws nothing.
    again = sub.submit_for_epoch("e_sub")
    assert again.success and not again.withdrawn
