"""Phase 7.1: Substrate.sol money ledger.

Money only (Decision 2026-07-10): Substrate.sol is a PURE MONEY contract.
Its reputation surface (agentReputation / reputationOfAt /
reputationTotalSupplyAt) is DELETED — REP lives DAO-side (RepToken),
claimed on ratified ATN earnings. ``agentMintTotal`` remains as the
cumulative tool-pool earnings counter (money). This file validates:

  1. agentMintTotal is monotonic: only ``claimFor`` can
     increase it, and it tracks the ATN minted.
  2. ATN balance is transferable via ``transfer``, ``approve`` /
     ``transferFrom``, and shaped like a minimal ERC20.
  3. Transferring ATN does not affect the cumulative earnings counter.
  4. atnTotalSupply tracks total ATN minted (== sum of all training
     records' amounts).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from nodes.common.canonical_ordering import canonical_order
from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import federated_epoch_close
from tests.substrate_chain_helpers import (
    deploy_substrate,
    register_bound,
    seed_pool,
)


SUBSTRATE_JSON = Path("C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json")


def _load_substrate() -> Tuple[list, str]:
    if not SUBSTRATE_JSON.exists():
        pytest.skip(f"missing artifact: {SUBSTRATE_JSON}")
    data = json.loads(SUBSTRATE_JSON.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _deploy(w3: Web3, deployer: str, abi: list, bytecode: str) -> str:
    # Genesis-mode anchor authority (account 9, registered).
    return deploy_substrate(w3, abi, bytecode, deployer)


def _register_agent(w3: Web3, contract, agent_addr: str, lineage_seed: str) -> None:
    peer_id = b"test-peer-" + lineage_seed.encode()
    receipt = register_bound(w3, contract, agent_addr, peer_id)
    assert receipt.status == 1, f"registerAgentBound failed: {receipt}"


@pytest.fixture
def chain():
    abi, bytecode = _load_substrate()
    w3 = Web3(EthereumTesterProvider())
    accounts = w3.eth.accounts
    deployer = accounts[0]
    addr = _deploy(w3, deployer, abi, bytecode)
    contract = w3.eth.contract(address=addr, abi=abi)

    agent_addrs = accounts[1:5]
    for i, ad in enumerate(agent_addrs):
        _register_agent(w3, contract, ad, f"agent-lineage-{i}")
    # Fees-only pool covering the test anchor's mints. The seed buys ATN
    # through the vault (accounts 6/7 hold it), so supply starts nonzero:
    # supply assertions below compare against this baseline.
    seed_pool(w3, contract)

    return {
        "w3": w3,
        "abi": abi,
        "addr": addr,
        "contract": contract,
        "deployer": deployer,
        "agent_addrs": list(agent_addrs),
    }


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _make_chain(rpb: str, kp: Keypair, agent_id: str, n: int = 2) -> List[EventBatch]:
    """v3 mint fixture (mint = tool usage only): each agent authors a
    pinned tool (digest derived from the agent id so agents don't
    collide), vetted by a distinct fleet and used by third-party
    attesting callers. Observation events alone no longer mint."""
    digest = hashlib.sha256(f"tool-{agent_id}".encode()).hexdigest()
    coords = [0.0] * (6 + 1024)
    coords[4] = 0.8
    coords[6 + 3] = 0.5
    reg = {
        "kind": "sub_claim_sprouted", "seq": 1,
        "author_agent": agent_id, "tendency_id": "correctness",
        "parent_id": "solver_root", "node_id": f"tm_{digest[:12]}",
        "position": "pro", "coords": list(coords),
        "polarity_axis": list(coords),
        "content": f"tool {digest[:8]}", "author_post": True,
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": "pinned", "author": agent_id},
    }

    def _vet(vetter, seq):
        return {
            "kind": "tool_used", "seq": seq, "author_agent": vetter,
            "manifest_digest": digest, "tool_author": agent_id,
            "receipt_digest": hashlib.sha256(
                f"vet-{agent_id}-{vetter}".encode()).hexdigest(),
            "ok": True, "fee_atn": 0.0, "vet": True,
        }

    def _receipt(caller, seq):
        return {
            "kind": "tool_used", "seq": seq, "author_agent": caller,
            "manifest_digest": digest, "tool_author": agent_id,
            "receipt_digest": hashlib.sha256(
                f"use-{agent_id}-{caller}".encode()).hexdigest(),
            "ok": True, "fee_atn": 0.0, "attested": True, "score": 0.8,
        }

    def _one(ev):
        return EventBatch(
            rpb_address=rpb, sender_pubkey=Keypair.generate().public_key,
            batch_seq=1, events=[ev], prev_batch_hash=b"",
            timestamp=1_700_000_000.0,
        )

    return [
        _one(reg),
        _one(_vet("vetter-1", 1)), _one(_vet("vetter-2", 1)),
        _one(_receipt("caller-1", 1)), _one(_receipt("caller-2", 2)),
    ]


def _build_close_and_anchor(chain_fixture, agent_ids, *, epoch_id: str = "e_71",
                             rpb: str = "rpb_71"):
    w3 = chain_fixture["w3"]
    deployer = chain_fixture["deployer"]
    senders = [Keypair.generate() for _ in agent_ids]
    batches: List[EventBatch] = []
    for kp, aid in zip(senders, agent_ids):
        batches.extend(_make_chain(rpb, kp, aid, n=2))
    canonical = canonical_order(batches)
    result = federated_epoch_close(canonical)
    result["epoch_id"] = epoch_id
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain_fixture["addr"]),
        web3=w3, contract_abi=chain_fixture["abi"],
    )
    anchor_result = anchorer.anchor_close_result(
        result, from_address=deployer)
    assert anchor_result.success
    resolver = InMemoryBlobResolver()
    cid = resolver.put(anchor_result.agent_mint_blob)
    assert cid == anchor_result.agent_mint_cid
    return result, epoch_id, resolver


# ---------------------------------------------------------------------------
# Earnings counter: monotonic, money-only
# ---------------------------------------------------------------------------


def test_reputation_surface_deleted_from_substrate(chain):
    """Substrate.sol is a pure money contract: the reputation view surface
    (agentReputation / reputationOfAt / reputationTotalSupplyAt) is GONE.
    agentMintTotal (money) remains."""
    abi = chain["abi"]
    fn_names = {f.get("name") for f in abi if f.get("type") == "function"}
    for gone in ("agentReputation", "reputationOfAt",
                 "reputationTotalSupplyAt"):
        assert gone not in fn_names, f"{gone} should be deleted"
    assert "agentMintTotal" in fn_names


def test_training_mints_atn_and_bumps_earnings(chain):
    """claimFor mints ATN to the claimant (not the sender) and increments
    agentMintTotal by the same amount; atnTotalSupply tracks it."""
    w3 = chain["w3"]
    contract = chain["contract"]
    addr = chain["agent_addrs"][0]
    # Agent ids ARE addresses: the mint leaf names the claimant (each
    # agent is its own household here, keyed lowercase). Claims are sent
    # by the deployer (an operator key), not the claimant.
    agent_ids = list(chain["agent_addrs"])

    earn_before = contract.functions.agentMintTotal(addr).call()
    bal_before = contract.functions.balanceOf(addr).call()
    supply_before = contract.functions.atnTotalSupply().call()
    # supply_before is the seed epoch's purchased ATN (nonzero); the
    # mint is asserted as a delta on top of it.
    assert earn_before == 0 and bal_before == 0

    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_money",
    )
    minting_aid = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid.lower(), 0) > 0),
        None,
    )
    assert minting_aid is not None, "no positive mint"
    minting_addr = chain["agent_addrs"][agent_ids.index(minting_aid)]

    submitter = AuthoritativeChainSubmitter(
        claimant_id=minting_aid.lower(), claimant_address=minting_addr,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        sender_address=chain["deployer"],
        web3=w3, substrate_abi=chain["abi"],
    )
    op_bal_before = contract.functions.balanceOf(chain["deployer"]).call()
    s = submitter.submit_for_epoch(epoch_id)
    assert s.success, s.error
    expected = s.contribution_scaled
    assert expected > 0
    # The operator paid gas but received no ATN.
    assert contract.functions.balanceOf(chain["deployer"]).call() == op_bal_before

    earn_after = contract.functions.agentMintTotal(minting_addr).call()
    bal_after = contract.functions.balanceOf(minting_addr).call()
    supply_after = contract.functions.atnTotalSupply().call()

    assert earn_after == expected
    assert bal_after == expected
    assert supply_after == supply_before + expected


# ---------------------------------------------------------------------------
# ATN: ERC20-shaped surface
# ---------------------------------------------------------------------------


def test_atn_transfer_moves_balance_does_not_affect_earnings(chain):
    """An agent transfers ATN to a peer. ATN balance moves; the
    cumulative earnings counter (agentMintTotal) on both sides unchanged."""
    w3 = chain["w3"]
    contract = chain["contract"]
    # Agent ids ARE addresses: the mint leaf names the claimant (each
    # agent is its own household here, keyed lowercase). Claims are sent
    # by the deployer (an operator key), not the claimant.
    agent_ids = list(chain["agent_addrs"])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_xfer",
    )
    src_aid = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid.lower(), 0) > 0),
        None,
    )
    assert src_aid is not None, "no positive mint"
    src_addr = chain["agent_addrs"][agent_ids.index(src_aid)]
    # dst is some other registered agent address.
    dst_addr = next(a for a in chain["agent_addrs"] if a != src_addr)

    submitter = AuthoritativeChainSubmitter(
        claimant_id=src_aid.lower(), claimant_address=src_addr,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        sender_address=chain["deployer"],
        web3=w3, substrate_abi=chain["abi"],
    )
    s = submitter.submit_for_epoch(epoch_id)
    assert s.success and s.contribution_scaled > 0

    src_rep_before = contract.functions.agentMintTotal(src_addr).call()
    src_bal_before = contract.functions.balanceOf(src_addr).call()
    dst_rep_before = contract.functions.agentMintTotal(dst_addr).call()
    dst_bal_before = contract.functions.balanceOf(dst_addr).call()

    transfer_amount = src_bal_before // 2
    assert transfer_amount > 0
    tx = contract.functions.transfer(dst_addr, transfer_amount).transact({
        "from": src_addr, "gas": 200_000,
    })
    rcpt = w3.eth.wait_for_transaction_receipt(tx)
    assert rcpt.status == 1

    src_rep_after = contract.functions.agentMintTotal(src_addr).call()
    src_bal_after = contract.functions.balanceOf(src_addr).call()
    dst_rep_after = contract.functions.agentMintTotal(dst_addr).call()
    dst_bal_after = contract.functions.balanceOf(dst_addr).call()

    # Balances moved.
    assert src_bal_after == src_bal_before - transfer_amount
    assert dst_bal_after == dst_bal_before + transfer_amount
    # Cumulative earnings unchanged on both sides.
    assert src_rep_after == src_rep_before
    assert dst_rep_after == dst_rep_before


def test_atn_transfer_to_zero_address_reverts(chain):
    w3 = chain["w3"]
    contract = chain["contract"]
    src = chain["agent_addrs"][0]

    rejected = False
    try:
        tx = contract.functions.transfer(
            "0x0000000000000000000000000000000000000000", 1,
        ).transact({"from": src, "gas": 200_000})
        rcpt = w3.eth.wait_for_transaction_receipt(tx)
        if rcpt.status != 1:
            rejected = True
    except Exception:
        rejected = True
    assert rejected


def test_atn_transfer_more_than_balance_reverts(chain):
    """Insufficient balance must revert (no overflow, no silent zeroing)."""
    w3 = chain["w3"]
    contract = chain["contract"]
    src = chain["agent_addrs"][0]
    dst = chain["agent_addrs"][1]

    # Source has zero balance (no training submitted).
    assert contract.functions.balanceOf(src).call() == 0
    rejected = False
    try:
        tx = contract.functions.transfer(dst, 100).transact({
            "from": src, "gas": 200_000,
        })
        rcpt = w3.eth.wait_for_transaction_receipt(tx)
        if rcpt.status != 1:
            rejected = True
    except Exception:
        rejected = True
    assert rejected


def test_atn_approve_and_transfer_from(chain):
    """Standard ERC20 approve / transferFrom flow."""
    w3 = chain["w3"]
    contract = chain["contract"]
    # Agent ids ARE addresses: the mint leaf names the claimant (each
    # agent is its own household here, keyed lowercase). Claims are sent
    # by the deployer (an operator key), not the claimant.
    agent_ids = list(chain["agent_addrs"])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_allow",
    )
    src_aid = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid.lower(), 0) > 0),
        None,
    )
    assert src_aid is not None, "no positive mint"
    src = chain["agent_addrs"][agent_ids.index(src_aid)]
    spender = next(a for a in chain["agent_addrs"] if a != src)
    dst = next(
        a for a in chain["agent_addrs"]
        if a != src and a != spender
    )

    submitter = AuthoritativeChainSubmitter(
        claimant_id=src_aid.lower(), claimant_address=src,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        sender_address=chain["deployer"],
        web3=w3, substrate_abi=chain["abi"],
    )
    submitter.submit_for_epoch(epoch_id)
    src_bal = contract.functions.balanceOf(src).call()
    assert src_bal > 0

    # approve
    spend_amt = src_bal // 4
    assert spend_amt > 0
    tx = contract.functions.approve(spender, spend_amt).transact({
        "from": src, "gas": 200_000,
    })
    w3.eth.wait_for_transaction_receipt(tx)
    assert contract.functions.allowance(src, spender).call() == spend_amt

    # transferFrom by spender
    tx = contract.functions.transferFrom(src, dst, spend_amt).transact({
        "from": spender, "gas": 200_000,
    })
    rcpt = w3.eth.wait_for_transaction_receipt(tx)
    assert rcpt.status == 1

    # Allowance decreased.
    assert contract.functions.allowance(src, spender).call() == 0
    # Balance moved from src to dst (NOT to spender).
    assert contract.functions.balanceOf(dst).call() == spend_amt


def test_atn_transfer_from_without_allowance_reverts(chain):
    w3 = chain["w3"]
    contract = chain["contract"]
    src = chain["agent_addrs"][0]
    spender = chain["agent_addrs"][1]
    dst = chain["agent_addrs"][2]

    rejected = False
    try:
        tx = contract.functions.transferFrom(src, dst, 1).transact({
            "from": spender, "gas": 200_000,
        })
        rcpt = w3.eth.wait_for_transaction_receipt(tx)
        if rcpt.status != 1:
            rejected = True
    except Exception:
        rejected = True
    assert rejected


def test_unlimited_approval_does_not_decrease(chain):
    """Setting allowance to MAX_UINT256 means transferFrom doesn't
    decrement it (matches ERC20 unlimited-approval convention)."""
    w3 = chain["w3"]
    contract = chain["contract"]
    # Agent ids ARE addresses: the mint leaf names the claimant (each
    # agent is its own household here, keyed lowercase). Claims are sent
    # by the deployer (an operator key), not the claimant.
    agent_ids = list(chain["agent_addrs"])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_unlim",
    )
    src_aid = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid.lower(), 0) > 0),
        None,
    )
    assert src_aid is not None, "no positive mint"
    src = chain["agent_addrs"][agent_ids.index(src_aid)]
    spender = next(a for a in chain["agent_addrs"] if a != src)
    dst = next(
        a for a in chain["agent_addrs"]
        if a != src and a != spender
    )

    submitter = AuthoritativeChainSubmitter(
        claimant_id=src_aid.lower(), claimant_address=src,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        sender_address=chain["deployer"],
        web3=w3, substrate_abi=chain["abi"],
    )
    submitter.submit_for_epoch(epoch_id)

    MAX_U256 = 2**256 - 1
    contract.functions.approve(spender, MAX_U256).transact({
        "from": src, "gas": 200_000,
    })

    src_bal = contract.functions.balanceOf(src).call()
    spend = min(src_bal, 1000)
    contract.functions.transferFrom(src, dst, spend).transact({
        "from": spender, "gas": 200_000,
    })
    # Allowance still MAX after transferFrom.
    assert contract.functions.allowance(src, spender).call() == MAX_U256


# ---------------------------------------------------------------------------
# Total supply
# ---------------------------------------------------------------------------


def test_atn_total_supply_tracks_minted(chain):
    """atnTotalSupply == sum of all training records' amounts.
    Transfers don't change supply."""
    w3 = chain["w3"]
    contract = chain["contract"]

    # Baseline: the seed epoch's purchased (vault-minted) ATN.
    supply0 = contract.functions.atnTotalSupply().call()

    # Agent ids ARE addresses: the mint leaf names the claimant (each
    # agent is its own household here, keyed lowercase). Claims are sent
    # by the deployer (an operator key), not the claimant.
    agent_ids = list(chain["agent_addrs"])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_supply",
    )
    expected_total = 0
    for aid, addr in zip(agent_ids, chain["agent_addrs"]):
        if result["agent_mint"].get(aid.lower(), 0) <= 0:
            continue
        sub = AuthoritativeChainSubmitter(
            claimant_id=aid.lower(), claimant_address=addr, resolver=resolver,
            config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
            sender_address=chain["deployer"],
            web3=w3, substrate_abi=chain["abi"],
        )
        s = sub.submit_for_epoch(epoch_id)
        if s.success and s.contribution_scaled > 0:
            expected_total += s.contribution_scaled

    assert expected_total > 0
    assert contract.functions.atnTotalSupply().call() == supply0 + expected_total

    # Now do a transfer; supply unchanged.
    if expected_total > 0:
        # Find an agent with positive balance.
        src = next(
            a for a in chain["agent_addrs"]
            if contract.functions.balanceOf(a).call() > 0
        )
        dst = next(a for a in chain["agent_addrs"] if a != src)
        bal = contract.functions.balanceOf(src).call()
        contract.functions.transfer(dst, bal // 2).transact({
            "from": src, "gas": 200_000,
        })
        assert contract.functions.atnTotalSupply().call() == supply0 + expected_total
