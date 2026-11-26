"""Governor-only anchoring + on-chain conservation (Decision 2026-10-16).

  - ``Substrate.anchor`` reverts for any sender but the governor; the
    governor's anchor lands (direct and through the real EpochAnchorer).
  - The hash chain still binds: a stale prevEpochRoot / prevAnchorHash
    reverts even from the governor, and an epoch anchors once.
  - split_pools: integer shares never exceed a pool, token by token, and
    the claim leaves built from them sum to <= the pool exactly.
  - An honest anchor at the seeded pools lets every leaf claim; the claims
    of each token sum to <= its pool cap.
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
from nodes.common.federated_reconcile import split_pools
from nodes.common.mint_merkle import claim_leaves, claim_merkle_proof
from tests.substrate_chain_helpers import (
    ZERO,
    anchor_direct,
    deploy_substrate,
    index_tokens,
    register_bound,
    seed_pool,
)

ARTIFACT = (Path(__file__).resolve().parent.parent / "artifacts" / "contracts"
            / "core" / "Substrate.sol" / "Substrate.json")


# ---------------------------------------------------------------------------
# Pure: integer flooring to the pool
# ---------------------------------------------------------------------------


def _claims(baskets):
    return {k: {"tokens": v, "value": 0} for k, v in baskets.items()}


@pytest.mark.parametrize("seed", range(20))
def test_split_never_exceeds_any_pool(seed):
    rng = random.Random(seed)
    t6, t18 = "0x" + "a6" * 20, "0x" + "1b" * 20
    pools = {t6: rng.randint(1, 10 ** 13), t18: rng.randint(1, 10 ** 30)}
    n = rng.randint(1, 40)
    weights = {Account.create().address: rng.randint(1, 10 ** 12) for _ in range(n)}
    leaves = claim_leaves(_claims(split_pools(pools, weights)))
    for t, pool in pools.items():
        paid = sum(a for _, toks, amts in leaves for tt, a in zip(toks, amts)
                   if tt.lower() == t)
        assert pool - n <= paid <= pool


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


def _close(claims, epoch_id):
    payload = {
        "epoch_root": "22" * 32, "claims": claims, "pools": {}, "parity": {},
        "total_value": 0, "output_decimals": 10,
        "n_batches": 1, "n_events": len(claims),
    }
    return {"authoritative_payload": payload, "claims": claims,
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
        _close({attacker: {"tokens": {"0x" + "a6" * 20: 10 ** 9}, "value": 0}}, "evil"),
        from_address=attacker)
    assert not evil.success
    assert c.functions.anchorCount().call() == 0
    ok = anchorer.anchor_close_result(
        _close({}, "honest"), from_address=chain["deployer"])
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


def test_split_close_claims_fully_and_conserves(chain):
    w3, c, anchorer = chain["w3"], chain["contract"], chain["anchorer"]
    pools = seed_pool(w3, c, gross=10 ** 9 + 7)
    agents = chain["agents"]
    claims = _claims(split_pools(pools, dict(zip(agents, [37, 111, 290]))))
    r = anchorer.anchor_close_result(
        _close(claims, "e-cons"), from_address=chain["deployer"])
    assert r.success, r.error
    tokens = index_tokens(c)
    for t in tokens:
        assert c.functions.poolCap(t.address).call() == pools[t.address.lower()]
    eid = epoch_id_hash("e-cons")
    for earner, toks, amts in claim_leaves(claims):
        proof = claim_merkle_proof(claims, earner)
        # claim is permissionless: the deployer (an operator key) sends every
        # claim; the same transaction pays the basket to the leaf's earner.
        by_addr = {t.address.lower(): t for t in tokens}
        before = {tt: by_addr[tt.lower()].functions.balanceOf(earner).call() for tt in toks}
        tx = c.functions.claim(earner, eid, toks, amts, proof).transact(
            {"from": chain["deployer"], "gas": 1_500_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        for tt, a in zip(toks, amts):
            assert c.functions.owed(earner, tt).call() == 0
            assert by_addr[tt.lower()].functions.balanceOf(earner).call() - before[tt] == a
    for t in tokens:
        assert c.functions.claimedTotal(t.address).call() <= pools[t.address.lower()]
