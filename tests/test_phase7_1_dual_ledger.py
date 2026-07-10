"""Phase 7.1: Substrate.sol money ledger.

Money only (Decision 2026-07-10): Substrate.sol is a PURE MONEY contract.
Its reputation surface (agentReputation / reputationOfAt /
reputationTotalSupplyAt) is DELETED — REP lives DAO-side (RepToken),
claimed on ratified ATN earnings. ``agentMintTotal`` remains as the
cumulative tool-pool earnings counter (money). This file validates:

  1. agentMintTotal is monotonic: only ``recordTrainingForEpoch`` can
     increase it, and it tracks the ATN minted.
  2. ATN balance is transferable via ``transfer``, ``approve`` /
     ``transferFrom``, and shaped like a minimal ERC20.
  3. Transferring ATN does not affect the cumulative earnings counter.
  4. atnTotalSupply tracks total ATN minted (== sum of all training
     records' amounts).
"""

from __future__ import annotations

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


SUBSTRATE_JSON = Path("C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json")


def _load_substrate() -> Tuple[list, str]:
    if not SUBSTRATE_JSON.exists():
        pytest.skip(f"missing artifact: {SUBSTRATE_JSON}")
    data = json.loads(SUBSTRATE_JSON.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _deploy(w3: Web3, deployer: str, abi: list, bytecode: str) -> str:
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    zero = "0x0000000000000000000000000000000000000000"
    tx = contract.constructor(deployer, zero, zero).transact({"from": deployer, "gas": 8_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    return receipt.contractAddress


def _register_agent(w3: Web3, contract, agent_addr: str, lineage_seed: str) -> None:
    lineage = Web3.keccak(text=lineage_seed)
    peer_id = b"test-peer-" + lineage_seed.encode()
    tx = contract.functions.registerAgent(lineage, peer_id).transact({
        "from": agent_addr, "gas": 300_000,
    })
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1


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
    chain: List[EventBatch] = []
    prev = b""
    for i in range(1, n + 1):
        coords = [0.0] * (4 + 1024)
        coords[2] = 0.5
        coords[4 + (10 * (i + 1))] = 0.5
        ev = {
            "kind": "observation_added",
            "seq": 1,
            "author_agent": agent_id,
            "obs_id": f"obs_{agent_id}_{i}",
            "coords": coords,
            "label": f"{agent_id}_{i}",
        }
        b = EventBatch(
            rpb_address=rpb, sender_pubkey=kp.public_key,
            batch_seq=i, events=[ev], prev_batch_hash=prev,
            timestamp=1_700_000_000.0 + i,
        )
        chain.append(b); prev = b.content_hash()
    return chain


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
    anchor_result = anchorer.anchor_close_result(result, from_address=deployer)
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
    """recordTrainingForEpoch mints ATN and increments agentMintTotal by
    the same amount; atnTotalSupply tracks it."""
    w3 = chain["w3"]
    contract = chain["contract"]
    addr = chain["agent_addrs"][0]
    # Agent ids ARE addresses: merkle mint proofs are keyed by
    # msg.sender, so claimable mint-map entries must be address-keyed.
    agent_ids = list(chain["agent_addrs"])

    earn_before = contract.functions.agentMintTotal(addr).call()
    bal_before = contract.functions.balanceOf(addr).call()
    supply_before = contract.functions.atnTotalSupply().call()
    assert earn_before == 0 and bal_before == 0 and supply_before == 0

    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_money",
    )
    minting_aid = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid, 0) > 0),
        None,
    )
    if minting_aid is None:
        pytest.skip("no positive mint")
    minting_addr = chain["agent_addrs"][agent_ids.index(minting_aid)]

    submitter = AuthoritativeChainSubmitter(
        agent_id=minting_aid, agent_address=minting_addr,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=w3, substrate_abi=chain["abi"],
    )
    s = submitter.submit_for_epoch(epoch_id)
    assert s.success
    expected = s.contribution_scaled

    earn_after = contract.functions.agentMintTotal(minting_addr).call()
    bal_after = contract.functions.balanceOf(minting_addr).call()
    supply_after = contract.functions.atnTotalSupply().call()

    assert earn_after == expected
    assert bal_after == expected
    assert supply_after == expected


# ---------------------------------------------------------------------------
# ATN: ERC20-shaped surface
# ---------------------------------------------------------------------------


def test_atn_transfer_moves_balance_does_not_affect_earnings(chain):
    """An agent transfers ATN to a peer. ATN balance moves; the
    cumulative earnings counter (agentMintTotal) on both sides unchanged."""
    w3 = chain["w3"]
    contract = chain["contract"]
    # Agent ids ARE addresses: merkle mint proofs are keyed by
    # msg.sender, so claimable mint-map entries must be address-keyed.
    agent_ids = list(chain["agent_addrs"])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_xfer",
    )
    src_aid = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid, 0) > 0),
        None,
    )
    if src_aid is None:
        pytest.skip("no positive mint")
    src_addr = chain["agent_addrs"][agent_ids.index(src_aid)]
    # dst is some other registered agent address.
    dst_addr = next(a for a in chain["agent_addrs"] if a != src_addr)

    submitter = AuthoritativeChainSubmitter(
        agent_id=src_aid, agent_address=src_addr,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
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
    # Agent ids ARE addresses: merkle mint proofs are keyed by
    # msg.sender, so claimable mint-map entries must be address-keyed.
    agent_ids = list(chain["agent_addrs"])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_allow",
    )
    src_aid = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid, 0) > 0),
        None,
    )
    if src_aid is None:
        pytest.skip("no positive mint")
    src = chain["agent_addrs"][agent_ids.index(src_aid)]
    spender = next(a for a in chain["agent_addrs"] if a != src)
    dst = next(
        a for a in chain["agent_addrs"]
        if a != src and a != spender
    )

    submitter = AuthoritativeChainSubmitter(
        agent_id=src_aid, agent_address=src,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
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
    # Agent ids ARE addresses: merkle mint proofs are keyed by
    # msg.sender, so claimable mint-map entries must be address-keyed.
    agent_ids = list(chain["agent_addrs"])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_unlim",
    )
    src_aid = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid, 0) > 0),
        None,
    )
    if src_aid is None:
        pytest.skip("no positive mint")
    src = chain["agent_addrs"][agent_ids.index(src_aid)]
    spender = next(a for a in chain["agent_addrs"] if a != src)
    dst = next(
        a for a in chain["agent_addrs"]
        if a != src and a != spender
    )

    submitter = AuthoritativeChainSubmitter(
        agent_id=src_aid, agent_address=src,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
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

    assert contract.functions.atnTotalSupply().call() == 0

    # Agent ids ARE addresses: merkle mint proofs are keyed by
    # msg.sender, so claimable mint-map entries must be address-keyed.
    agent_ids = list(chain["agent_addrs"])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_71_supply",
    )
    expected_total = 0
    for aid, addr in zip(agent_ids, chain["agent_addrs"]):
        if result["agent_mint"].get(aid, 0) <= 0:
            continue
        sub = AuthoritativeChainSubmitter(
            agent_id=aid, agent_address=addr, resolver=resolver,
            config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
            web3=w3, substrate_abi=chain["abi"],
        )
        s = sub.submit_for_epoch(epoch_id)
        if s.success and s.contribution_scaled > 0:
            expected_total += s.contribution_scaled

    assert contract.functions.atnTotalSupply().call() == expected_total

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
        assert contract.functions.atnTotalSupply().call() == expected_total
