"""Phase 7.2: payForInference — structured ATN transfer for inference fees.

Validates:

  1. Happy path: caller pays recipient; balances move; both ATNTransfer
     and InferencePayment events fire with correct fields.
  2. requestId is propagated as the third indexed topic on the event,
     so off-chain auditors can match payments to requests.
  3. Insufficient balance reverts; no state change.
  4. Recipient = zero address reverts (no value gets stuck).
  5. Reputation untouched on both sides — payments are pure ATN flows.
  6. atnTotalSupply unchanged by payments (it's still a transfer, not
     a mint or burn).
  7. Multiple payments accumulate cleanly on the recipient.
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
    tx = contract.constructor().transact({"from": deployer, "gas": 8_000_000})
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


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _make_chain(rpb: str, agent_id: str) -> List[EventBatch]:
    """v3 mint fixture (mint = tool usage only): the agent authors a
    pinned tool that gets vetted (distinct-fleet greenlight) and used
    by third-party attesting callers. Observation events alone no
    longer mint anything."""
    digest = "72" * 32
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
            "receipt_digest": f"v{seq:02d}" * 8, "ok": True,
            "fee_atn": 0.0, "vet": True,
        }
    def _receipt(caller, seq):
        return {
            "kind": "tool_used", "seq": seq, "author_agent": caller,
            "manifest_digest": digest, "tool_author": agent_id,
            "receipt_digest": f"r{seq:02d}" * 8, "ok": True,
            "fee_atn": 0.0, "attested": True, "score": 0.8,
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


def _give_agent_some_atn(chain_fixture, agent_id: str, agent_addr: str) -> int:
    """Run a federated close + anchor + submit so the agent gets ATN.
    Returns the ATN balance the agent now holds."""
    w3 = chain_fixture["w3"]
    deployer = chain_fixture["deployer"]
    rpb = "rpb_72_seed"
    batches = _make_chain(rpb, agent_id)
    canonical = canonical_order(batches)
    result = federated_epoch_close(canonical)
    epoch_id = "e_72_seed"
    result["epoch_id"] = epoch_id

    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain_fixture["addr"]),
        web3=w3, contract_abi=chain_fixture["abi"],
    )
    s = anchorer.anchor_close_result(result, from_address=deployer)
    assert s.success
    resolver = InMemoryBlobResolver()
    resolver.put(s.agent_mint_blob)

    submitter = AuthoritativeChainSubmitter(
        agent_id=agent_id, agent_address=agent_addr, resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain_fixture["addr"]),
        web3=w3, substrate_abi=chain_fixture["abi"],
    )
    sub = submitter.submit_for_epoch(epoch_id)
    assert sub.success and sub.contribution_scaled > 0
    return chain_fixture["contract"].functions.balanceOf(agent_addr).call()


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
        "w3": w3, "abi": abi, "addr": addr, "contract": contract,
        "deployer": deployer, "agent_addrs": list(agent_addrs),
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_pay_for_inference_moves_balance_and_emits_event(chain):
    w3 = chain["w3"]
    contract = chain["contract"]
    payer_addr = chain["agent_addrs"][0]
    recipient = chain["agent_addrs"][1]

    payer_initial = _give_agent_some_atn(chain, payer_addr, payer_addr)
    assert payer_initial > 0
    recipient_before = contract.functions.balanceOf(recipient).call()
    supply_before = contract.functions.atnTotalSupply().call()

    pay_amount = payer_initial // 4
    request_id = Web3.keccak(text="inference-request-abc123")

    tx = contract.functions.payForInference(
        recipient, pay_amount, request_id,
    ).transact({"from": payer_addr, "gas": 200_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1

    payer_after = contract.functions.balanceOf(payer_addr).call()
    recipient_after = contract.functions.balanceOf(recipient).call()
    supply_after = contract.functions.atnTotalSupply().call()

    assert payer_after == payer_initial - pay_amount
    assert recipient_after == recipient_before + pay_amount
    assert supply_after == supply_before  # transfer, not mint/burn

    # InferencePayment event with the right fields.
    inf_events = contract.events.InferencePayment().process_receipt(receipt)
    assert len(inf_events) == 1
    args = inf_events[0]["args"]
    assert args["payer"] == payer_addr
    assert args["recipient"] == recipient
    assert args["amount"] == pay_amount
    assert args["requestId"] == request_id

    # ATNTransfer event also fires.
    xfer_events = contract.events.ATNTransfer().process_receipt(receipt)
    assert len(xfer_events) == 1
    xfer = xfer_events[0]["args"]
    assert xfer["from"] == payer_addr
    assert xfer["to"] == recipient
    assert xfer["amount"] == pay_amount


def test_request_id_indexed_for_filtering(chain):
    """The requestId is indexed so off-chain auditors can filter
    InferencePayment logs by request id."""
    abi = chain["abi"]
    inf_event = next(
        e for e in abi
        if e.get("type") == "event" and e.get("name") == "InferencePayment"
    )
    req_param = next(p for p in inf_event["inputs"] if p["name"] == "requestId")
    assert req_param["indexed"] is True


def test_multiple_payments_accumulate_on_recipient(chain):
    w3 = chain["w3"]
    contract = chain["contract"]
    payer_addr = chain["agent_addrs"][0]
    recipient = chain["agent_addrs"][1]

    payer_initial = _give_agent_some_atn(chain, payer_addr, payer_addr)
    recipient_before = contract.functions.balanceOf(recipient).call()

    pay_each = payer_initial // 8
    total_paid = 0
    for i in range(3):
        rid = Web3.keccak(text=f"req-{i}")
        tx = contract.functions.payForInference(
            recipient, pay_each, rid,
        ).transact({"from": payer_addr, "gas": 200_000})
        receipt = w3.eth.wait_for_transaction_receipt(tx)
        assert receipt.status == 1
        total_paid += pay_each

    assert (
        contract.functions.balanceOf(recipient).call()
        == recipient_before + total_paid
    )


# ---------------------------------------------------------------------------
# Reverts
# ---------------------------------------------------------------------------


def test_pay_for_inference_with_insufficient_balance_reverts(chain):
    w3 = chain["w3"]
    contract = chain["contract"]
    # agent_addrs[2] hasn't received any training mint -> zero balance.
    poor_payer = chain["agent_addrs"][2]
    recipient = chain["agent_addrs"][3]
    assert contract.functions.balanceOf(poor_payer).call() == 0

    rid = Web3.keccak(text="rejected-tx")
    rejected = False
    try:
        tx = contract.functions.payForInference(
            recipient, 100, rid,
        ).transact({"from": poor_payer, "gas": 200_000})
        rcpt = w3.eth.wait_for_transaction_receipt(tx)
        if rcpt.status != 1:
            rejected = True
    except Exception:
        rejected = True
    assert rejected
    # State unchanged.
    assert contract.functions.balanceOf(poor_payer).call() == 0
    assert contract.functions.balanceOf(recipient).call() == 0


def test_pay_to_zero_address_reverts(chain):
    w3 = chain["w3"]
    contract = chain["contract"]
    payer_addr = chain["agent_addrs"][0]
    _give_agent_some_atn(chain, payer_addr, payer_addr)

    rid = Web3.keccak(text="zero-target")
    rejected = False
    try:
        tx = contract.functions.payForInference(
            "0x0000000000000000000000000000000000000000", 1, rid,
        ).transact({"from": payer_addr, "gas": 200_000})
        rcpt = w3.eth.wait_for_transaction_receipt(tx)
        if rcpt.status != 1:
            rejected = True
    except Exception:
        rejected = True
    assert rejected


# ---------------------------------------------------------------------------
# Reputation isolation
# ---------------------------------------------------------------------------


def test_payment_does_not_change_reputation_on_either_side(chain):
    w3 = chain["w3"]
    contract = chain["contract"]
    payer_addr = chain["agent_addrs"][0]
    recipient = chain["agent_addrs"][1]

    _give_agent_some_atn(chain, payer_addr, payer_addr)

    payer_rep_before = contract.functions.agentReputation(payer_addr).call()
    recip_rep_before = contract.functions.agentReputation(recipient).call()
    # payer earned reputation from training; recipient hasn't.
    assert payer_rep_before > 0
    assert recip_rep_before == 0

    pay_amount = contract.functions.balanceOf(payer_addr).call() // 2
    rid = Web3.keccak(text="rep-isolation")
    tx = contract.functions.payForInference(
        recipient, pay_amount, rid,
    ).transact({"from": payer_addr, "gas": 200_000})
    w3.eth.wait_for_transaction_receipt(tx)

    payer_rep_after = contract.functions.agentReputation(payer_addr).call()
    recip_rep_after = contract.functions.agentReputation(recipient).call()
    # Neither side's reputation moved. ATN is the spending tier;
    # reputation is the contribution tier.
    assert payer_rep_after == payer_rep_before
    assert recip_rep_after == recip_rep_before


# ---------------------------------------------------------------------------
# Recipient doesn't have to be a registered agent
# ---------------------------------------------------------------------------


def test_payment_to_unregistered_address_works(chain):
    """A daemon serving inference doesn't need to be a registered
    *agent* — they just need a valid address that can hold ATN.
    Registration is only required for training-mint participation."""
    w3 = chain["w3"]
    contract = chain["contract"]
    payer_addr = chain["agent_addrs"][0]
    # accounts[5+] aren't registered as agents but exist on the chain.
    unregistered_recipient = w3.eth.accounts[5]
    assert not contract.functions.isRegistered(unregistered_recipient).call()

    _give_agent_some_atn(chain, payer_addr, payer_addr)
    pay_amount = 12345
    rid = Web3.keccak(text="anon-recipient")

    tx = contract.functions.payForInference(
        unregistered_recipient, pay_amount, rid,
    ).transact({"from": payer_addr, "gas": 200_000})
    rcpt = w3.eth.wait_for_transaction_receipt(tx)
    assert rcpt.status == 1
    assert contract.functions.balanceOf(unregistered_recipient).call() == pay_amount


# ---------------------------------------------------------------------------
# Self-payment (edge case worth pinning down)
# ---------------------------------------------------------------------------


def test_self_payment_is_a_no_op_in_balance_terms(chain):
    """payForInference(self, ...) shouldn't crash and shouldn't
    double-count. End balance unchanged."""
    w3 = chain["w3"]
    contract = chain["contract"]
    payer_addr = chain["agent_addrs"][0]
    initial = _give_agent_some_atn(chain, payer_addr, payer_addr)

    rid = Web3.keccak(text="self-pay")
    pay = initial // 4
    tx = contract.functions.payForInference(
        payer_addr, pay, rid,
    ).transact({"from": payer_addr, "gas": 200_000})
    rcpt = w3.eth.wait_for_transaction_receipt(tx)
    assert rcpt.status == 1
    # Balance unchanged: subtract then add same amount.
    assert contract.functions.balanceOf(payer_addr).call() == initial
