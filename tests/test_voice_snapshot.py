"""Checkpointed ATN + snapshot-pinned voice state (chain side).

Validates the IVotes-mechanism checkpoints on Substrate.sol
(``balanceOfAt`` / ``atnTotalSupplyAt`` — Trace208 history, no
delegation layer) and ``read_voice_state``'s architecture guarantee:
voices are priced from balances AS OF the previous epoch's anchor
block, so activity after the anchor cannot change this epoch's
weights, and any two daemons reading at different times derive the
identical maps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pytest

eth_tester = pytest.importorskip("eth_tester")

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from nodes.common.authoritative_submitter import (
    AuthoritativeChainSubmitter,
    AuthoritativeSubmitterConfig,
)
from nodes.common.blob_resolver import InMemoryBlobResolver
from nodes.common.canonical_ordering import canonical_order
from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import (
    BASE_EMISSION_PER_EPOCH,
    VOICE_EPSILON,
    federated_epoch_close,
)
from nodes.common.voice_state import read_voice_state

SUBSTRATE_JSON = Path(
    "C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json")


def _load_substrate() -> Tuple[list, str]:
    if not SUBSTRATE_JSON.exists():
        pytest.skip(f"missing artifact: {SUBSTRATE_JSON}")
    data = json.loads(SUBSTRATE_JSON.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _deploy(w3: Web3, deployer: str, abi: list, bytecode: str) -> str:
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor(deployer).transact({"from": deployer, "gas": 8_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    return receipt.contractAddress


def _register_agent(w3: Web3, contract, agent_addr: str, seed: str) -> None:
    lineage = Web3.keccak(text=seed)
    tx = contract.functions.registerAgent(
        lineage, b"peer-" + seed.encode()).transact({
            "from": agent_addr, "gas": 300_000})
    assert w3.eth.wait_for_transaction_receipt(tx).status == 1


_DIGEST = "ef" * 32


def _coords6() -> List[float]:
    out = [0.0] * (6 + 1024)
    out[4] = 0.8
    out[6 + 3] = 0.5
    return out


def _one_batch(rpb: str, ev: dict, kp: Keypair, seq: int) -> EventBatch:
    return EventBatch(
        rpb_address=rpb, sender_pubkey=kp.public_key, batch_seq=seq,
        events=[ev], prev_batch_hash=b"",
        timestamp=1_700_000_000.0 + seq,
    )


def _tool_batches(rpb: str, author_addr: str,
                  digest: str = _DIGEST) -> List[EventBatch]:
    """v3 mint fixture: mint is TOOL-USAGE only, so the epoch needs a
    registration (author = the chain agent address), a distinct-fleet
    greenlight, and attested receipts from third-party callers."""
    reg = {
        "kind": "sub_claim_sprouted", "seq": 1,
        "author_agent": author_addr, "tendency_id": "correctness",
        "parent_id": "solver_root", "node_id": f"tm_{digest[:12]}",
        "position": "pro", "coords": _coords6(),
        "polarity_axis": _coords6(),
        "content": f"tool {digest[:8]}", "author_post": True,
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": "pinned", "author": author_addr},
    }
    def _vet(vetter, seq):
        return {
            "kind": "tool_used", "seq": seq, "author_agent": vetter,
            "manifest_digest": digest, "tool_author": author_addr,
            "receipt_digest": f"v{seq:02d}" * 8, "ok": True,
            "fee_atn": 0.0, "vet": True,
        }
    def _receipt(caller, seq):
        return {
            "kind": "tool_used", "seq": seq, "author_agent": caller,
            "manifest_digest": digest, "tool_author": author_addr,
            "receipt_digest": f"r{seq:02d}" * 8, "ok": True,
            "fee_atn": 0.0, "attested": True, "score": 0.8,
        }
    return [
        _one_batch(rpb, reg, Keypair.generate(), 1),
        _one_batch(rpb, _vet("vetter-1", 1), Keypair.generate(), 1),
        _one_batch(rpb, _vet("vetter-2", 1), Keypair.generate(), 1),
        _one_batch(rpb, _receipt("caller-1", 1), Keypair.generate(), 1),
        _one_batch(rpb, _receipt("caller-2", 2), Keypair.generate(), 1),
    ]


def _anchor_epoch(fx, agent_ids, epoch_id: str, rpb: str):
    """Close + anchor an epoch (tool mint to agent_ids[0]); returns
    (close_result, resolver)."""
    batches = _tool_batches(rpb, agent_ids[0], digest=_DIGEST)
    result = federated_epoch_close(canonical_order(batches))
    result["epoch_id"] = epoch_id
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=fx["addr"]),
        web3=fx["w3"], contract_abi=fx["abi"],
    )
    anchor_result = anchorer.anchor_close_result(
        result, from_address=fx["deployer"])
    assert anchor_result.success
    resolver = InMemoryBlobResolver()
    resolver.put(anchor_result.agent_mint_blob)
    return result, resolver


def _mint_for(fx, result, resolver, epoch_id: str) -> Tuple[str, int]:
    """Submit recordTrainingForEpoch for the first minting agent;
    returns (address, scaled amount)."""
    aid = next(a for a in fx["agent_addrs"]
               if result["agent_mint"].get(a, 0) > 0)
    submitter = AuthoritativeChainSubmitter(
        agent_id=aid, agent_address=aid, resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=fx["addr"]),
        web3=fx["w3"], substrate_abi=fx["abi"],
    )
    s = submitter.submit_for_epoch(epoch_id)
    assert s.success
    return aid, s.contribution_scaled


@pytest.fixture
def fx():
    abi, bytecode = _load_substrate()
    w3 = Web3(EthereumTesterProvider())
    accounts = w3.eth.accounts
    deployer = accounts[0]
    addr = _deploy(w3, deployer, abi, bytecode)
    contract = w3.eth.contract(address=addr, abi=abi)
    agent_addrs = accounts[1:4]
    for i, ad in enumerate(agent_addrs):
        _register_agent(w3, contract, ad, f"voice-agent-{i}")
    return {
        "w3": w3, "abi": abi, "addr": addr, "contract": contract,
        "deployer": deployer, "agent_addrs": list(agent_addrs),
    }


class TestCheckpointedATN:
    def test_balance_of_at_tracks_history(self, fx):
        """Mint, then transfer: the pre-transfer block still reads the
        pre-transfer balance; latest reads the post-transfer one."""
        w3, contract = fx["w3"], fx["contract"]
        result, resolver = _anchor_epoch(
            fx, fx["agent_addrs"], "e_ckpt", "rpb_ckpt")
        minter, amount = _mint_for(fx, result, resolver, "e_ckpt")
        mint_block = w3.eth.block_number

        other = next(a for a in fx["agent_addrs"] if a != minter)
        tx = contract.functions.transfer(other, amount // 2).transact(
            {"from": minter, "gas": 200_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1

        # History: at the mint block, the minter still held everything.
        assert contract.functions.balanceOfAt(
            minter, mint_block).call() == amount
        assert contract.functions.balanceOfAt(other, mint_block).call() == 0
        # Present: split.
        assert contract.functions.balanceOf(minter).call() == (
            amount - amount // 2)
        assert contract.functions.balanceOf(other).call() == amount // 2
        # Before anything existed: zero (no checkpoint <= block).
        assert contract.functions.balanceOfAt(minter, 0).call() == 0
        # Supply history: zero before the mint, `amount` after.
        assert contract.functions.atnTotalSupplyAt(0).call() == 0
        assert contract.functions.atnTotalSupplyAt(
            mint_block).call() == amount
        assert contract.functions.atnTotalSupply().call() == amount


class TestVoiceStateSnapshot:
    def test_no_anchor_means_no_weights(self, fx):
        state = read_voice_state(fx["addr"], web3=fx["w3"])
        assert state["snapshot_block"] is None
        assert state["voice_weights"] == {}
        assert state["owner_map"] == {}
        # Epoch 1: floor-only pool (the faucet), nothing recycled yet.
        assert state["emission_pool"] == BASE_EMISSION_PER_EPOCH
        assert state["recycled"] == 0.0

    def test_weights_pinned_to_anchor_block(self, fx):
        """The architecture guarantee: activity AFTER the anchor cannot
        change this epoch's voice weights — two daemons reading at
        different times see the identical maps."""
        w3, contract = fx["w3"], fx["contract"]
        # Epoch 1: anchor, then mint (mint lands AFTER anchor #1's
        # block, so it's invisible to a snapshot at anchor #1).
        r1, res1 = _anchor_epoch(fx, fx["agent_addrs"], "e_v1", "rpb_v1")
        minter, amount = _mint_for(fx, r1, res1, "e_v1")

        state_1 = read_voice_state(fx["addr"], web3=fx["w3"])
        assert state_1["snapshot_block"] is not None
        assert state_1["supply"] == 0                      # pre-mint snapshot
        # All registered agents present, all at the epsilon floor.
        for a in fx["agent_addrs"]:
            assert state_1["voice_weights"][a.lower()] == pytest.approx(
                VOICE_EPSILON)

        # Epoch 2: anchor again — the new snapshot sees the mint.
        _anchor_epoch(fx, fx["agent_addrs"], "e_v2", "rpb_v2")
        state_2 = read_voice_state(fx["addr"], web3=fx["w3"])
        assert state_2["snapshot_block"] > state_1["snapshot_block"]
        assert state_2["supply"] == amount
        assert state_2["voice_weights"][minter.lower()] == pytest.approx(
            VOICE_EPSILON + 1.0)     # holds the entire supply

        # THE PIN: transfer after anchor #2 — a re-read still returns
        # the identical maps (mid-epoch funding carries no weight).
        other = next(a for a in fx["agent_addrs"] if a != minter)
        tx = contract.functions.transfer(other, amount).transact(
            {"from": minter, "gas": 200_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        state_2b = read_voice_state(fx["addr"], web3=fx["w3"])
        assert state_2b["voice_weights"] == state_2["voice_weights"]
        assert state_2b["snapshot_block"] == state_2["snapshot_block"]


class TestFeeRecycledEmission:
    def test_burned_fees_enter_exactly_one_window(self, fx):
        """A service payment's burned fee share raises the pool for the
        FIRST close whose snapshot window contains it, then leaves —
        recycling conserves; nothing is counted twice."""
        w3, contract = fx["w3"], fx["contract"]
        # Epoch 1: anchor + mint so a payer has ATN.
        r1, res1 = _anchor_epoch(fx, fx["agent_addrs"], "e_f1", "rpb_f1")
        payer, amount = _mint_for(fx, r1, res1, "e_f1")

        # Pay for a service — the fee burns + treasury share moves.
        other = next(a for a in fx["agent_addrs"] if a != payer)
        supply_before = contract.functions.atnTotalSupply().call()
        pay = amount // 2
        fee = pay * 250 // 10_000
        to_treasury = fee * 5_000 // 10_000
        burned = fee - to_treasury
        tx = contract.functions.payForInference(
            other, pay, Web3.keccak(text="svc-req")).transact(
            {"from": payer, "gas": 300_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        assert (contract.functions.atnTotalSupply().call()
                == supply_before - burned)

        # Anchor epoch 2 — its window contains the payment.
        _anchor_epoch(fx, fx["agent_addrs"], "e_f2", "rpb_f2")
        state = read_voice_state(fx["addr"], web3=fx["w3"])
        assert state["recycled"] == pytest.approx(burned / 1_000_000.0)
        assert state["emission_pool"] == pytest.approx(
            BASE_EMISSION_PER_EPOCH + burned / 1_000_000.0)

        # Anchor epoch 3 with no payments — the fee has left the
        # window; pool falls back to the floor.
        _anchor_epoch(fx, fx["agent_addrs"], "e_f3", "rpb_f3")
        state_3 = read_voice_state(fx["addr"], web3=fx["w3"])
        assert state_3["recycled"] == 0.0
        assert state_3["emission_pool"] == BASE_EMISSION_PER_EPOCH
