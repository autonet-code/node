"""Checkpointed ATN (money) + RepToken-sourced voice snapshot.

Money only on Substrate (Decision 2026-07-10): Substrate.sol keeps its
ATN checkpoints (``balanceOfAt`` / ``atnTotalSupplyAt``) but its
reputation surface is DELETED. Voice now reads REP share from RepToken
(DAO) pinned to the previous epoch's anchor TIMESTAMP (ERC20Votes,
mode=timestamp). ``read_voice_state`` guarantees: voices are priced AS OF
the previous anchor, so activity after it cannot change this epoch's
weights, and any two daemons reading at different times derive the
identical maps. Emission pool is FEES-ONLY (no base floor): zero service
volume => zero pool.

The headline money/voice split still holds: a vault ATN mint moves
balances but NOT voice weights (REP, not ATN, is voice).
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
    VOICE_EPSILON,
    federated_epoch_close,
)
from nodes.common.voice_state import read_voice_state

SUBSTRATE_JSON = Path(
    "C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json")
REP_JSON = Path(
    "C:/code/autonet/artifacts/contracts/test/MockRepToken.sol/MockRepToken.json")

_ZERO = "0x0000000000000000000000000000000000000000"


def _load_substrate() -> Tuple[list, str]:
    if not SUBSTRATE_JSON.exists():
        pytest.skip(f"missing artifact: {SUBSTRATE_JSON}")
    data = json.loads(SUBSTRATE_JSON.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _load_rep() -> Tuple[list, str]:
    if not REP_JSON.exists():
        pytest.skip(f"missing artifact: {REP_JSON}")
    data = json.loads(REP_JSON.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _deploy(w3: Web3, deployer: str, abi: list, bytecode: str,
            vault_minter: str = _ZERO) -> str:
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    # Substrate(treasury, vaultMinter, governor). governor=zero => frozen.
    tx = contract.constructor(deployer, vault_minter, _ZERO).transact(
        {"from": deployer, "gas": 8_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    return receipt.contractAddress


def _deploy_rep(w3: Web3, deployer: str) -> Tuple[str, list]:
    abi, bytecode = _load_rep()
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor().transact({"from": deployer, "gas": 3_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    return receipt.contractAddress, abi


def _set_votes(w3: Web3, rep_contract, deployer: str, account: str,
               value: int) -> None:
    tx = rep_contract.functions.setVotes(account, value).transact(
        {"from": deployer, "gas": 300_000})
    assert w3.eth.wait_for_transaction_receipt(tx).status == 1


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
    """v3 mint fixture: mint is TOOL-USAGE only — a registration (author =
    the chain agent address) plus attested receipts from third-party
    callers. voice_weights=None here (local regime), so mint is per-fleet
    log1p; the on-chain ATN it settles is what these tests read back."""
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
    def _receipt(caller, seq):
        return {
            "kind": "tool_used", "seq": seq, "author_agent": caller,
            "manifest_digest": digest, "tool_author": author_addr,
            "receipt_digest": f"r{seq:02d}" * 8, "ok": True,
            "fee_atn": 0.0, "attested": True, "score": 0.8,
        }
    return [
        _one_batch(rpb, reg, Keypair.generate(), 1),
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
    rep_addr, rep_abi = _deploy_rep(w3, deployer)
    rep_contract = w3.eth.contract(address=rep_addr, abi=rep_abi)
    return {
        "w3": w3, "abi": abi, "addr": addr, "contract": contract,
        "deployer": deployer, "agent_addrs": list(agent_addrs),
        "rep_addr": rep_addr, "rep_contract": rep_contract,
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
        state = read_voice_state(
            fx["addr"], web3=fx["w3"], rep_token_address=fx["rep_addr"])
        assert state["snapshot_block"] is None
        assert state["voice_weights"] == {}
        assert state["owner_map"] == {}
        # Fees-only: empty pool at genesis (no base floor).
        assert state["emission_pool"] == 0.0
        assert state["recycled"] == 0.0

    def test_no_rep_token_means_genesis_regime(self, fx):
        """No RepToken configured => empty rep maps (drift weight 1.0),
        pool still fees-only."""
        # An anchor exists so we exercise the non-cold-start path.
        _anchor_epoch(fx, fx["agent_addrs"], "e_g1", "rpb_g1")
        state = read_voice_state(fx["addr"], web3=fx["w3"])  # no rep token
        assert state["snapshot_block"] is not None
        assert state["voice_weights"] == {}
        assert state["rep_shares"] == {}
        assert state["rep_supply"] == 0

    def test_weights_from_reptoken_pinned_to_anchor(self, fx):
        """Voice reads REP from RepToken pinned to the anchor timestamp.
        Set votes BEFORE the anchor so the anchor-ts read sees them; a
        change AFTER the anchor does not move this epoch's weights."""
        w3 = fx["w3"]
        agents = fx["agent_addrs"]
        holder = agents[0]
        # Give the holder all the REP, then anchor (anchor ts >= the
        # setVotes ts, so getPastVotes(holder, anchor_ts) == the value).
        _set_votes(w3, fx["rep_contract"], fx["deployer"], holder, 1000)
        _anchor_epoch(fx, agents, "e_v1", "rpb_v1")

        state_1 = read_voice_state(
            fx["addr"], web3=w3, rep_token_address=fx["rep_addr"])
        assert state_1["snapshot_block"] is not None
        assert state_1["weight_source"] == "reputation"
        assert state_1["supply"] == 1000
        # Holder owns the entire REP supply => share 1.0 + epsilon floor.
        assert state_1["voice_weights"][holder.lower()] == pytest.approx(
            VOICE_EPSILON + 1.0)
        assert state_1["rep_shares"][holder.lower()] == pytest.approx(1.0)

        # THE PIN: change votes AFTER the anchor — a re-read against the
        # SAME anchor returns the identical maps.
        _set_votes(w3, fx["rep_contract"], fx["deployer"], agents[1], 9000)
        state_1b = read_voice_state(
            fx["addr"], web3=w3, rep_token_address=fx["rep_addr"])
        assert state_1b["voice_weights"] == state_1["voice_weights"]
        assert state_1b["snapshot_block"] == state_1["snapshot_block"]

    def test_owner_wallet_not_agent_has_no_rep_term(self, fx):
        """Only registered agents contribute REP to a household; a bare
        owner wallet (not an agent, no OwnerBound) appears in no
        household."""
        w3 = fx["w3"]
        agents = fx["agent_addrs"]
        owner_wallet = w3.eth.accounts[6]     # NOT a registered agent
        _set_votes(w3, fx["rep_contract"], fx["deployer"], agents[0], 500)
        # Even if the owner wallet somehow held votes, it's not an agent.
        _set_votes(w3, fx["rep_contract"], fx["deployer"], owner_wallet, 500)
        _anchor_epoch(fx, agents, "e_ow1", "rpb_ow1")

        state = read_voice_state(
            fx["addr"], web3=w3, rep_token_address=fx["rep_addr"])
        assert owner_wallet.lower() not in state["voice_weights"]
        # agents[0] holds 500 of the 1000 supply => share 0.5.
        assert state["rep_shares"][agents[0].lower()] == pytest.approx(0.5)


class TestFeeRecycledEmission:
    def test_burned_fees_enter_exactly_one_window(self, fx):
        """A service payment's burned fee share raises the fees-only pool
        for the FIRST close whose snapshot window contains it, then leaves.
        No base floor — the pool is exactly the recycled burn."""
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
        tx = contract.functions.payForService(
            other, pay, Web3.keccak(text="svc-req")).transact(
            {"from": payer, "gas": 300_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        assert (contract.functions.atnTotalSupply().call()
                == supply_before - burned)

        # Anchor epoch 2 — its window contains the payment. Pool == burn.
        _anchor_epoch(fx, fx["agent_addrs"], "e_f2", "rpb_f2")
        state = read_voice_state(
            fx["addr"], web3=fx["w3"], rep_token_address=fx["rep_addr"])
        assert state["recycled"] == pytest.approx(burned / 1_000_000.0)
        assert state["emission_pool"] == pytest.approx(burned / 1_000_000.0)

        # Anchor epoch 3 with no payments — the fee has left the window;
        # fees-only pool falls to zero (no base floor).
        _anchor_epoch(fx, fx["agent_addrs"], "e_f3", "rpb_f3")
        state_3 = read_voice_state(
            fx["addr"], web3=fx["w3"], rep_token_address=fx["rep_addr"])
        assert state_3["recycled"] == 0.0
        assert state_3["emission_pool"] == 0.0


class TestVaultMint:
    """mintFromVault: purchased ATN. Mints balance/supply/checkpoint like
    the training path (money), only vaultMinter may call, and a zero-vault
    deploy always reverts. Voice (REP) is a separate DAO ledger, untouched."""

    def test_vault_minter_mints_atn(self):
        abi, bytecode = _load_substrate()
        w3 = Web3(EthereumTesterProvider())
        accounts = w3.eth.accounts
        deployer, vault, buyer = accounts[0], accounts[1], accounts[2]
        addr = _deploy(w3, deployer, abi, bytecode, vault_minter=vault)
        contract = w3.eth.contract(address=addr, abi=abi)

        assert contract.functions.vaultMinter().call() == vault
        supply_before = contract.functions.atnTotalSupply().call()
        bal_before = contract.functions.balanceOf(buyer).call()

        amount = 7_000
        tx = contract.functions.mintFromVault(buyer, amount).transact(
            {"from": vault, "gas": 300_000})
        receipt = w3.eth.wait_for_transaction_receipt(tx)
        assert receipt.status == 1
        mint_block = w3.eth.block_number

        # Money moved: balance + supply + checkpoints.
        assert contract.functions.balanceOf(buyer).call() == bal_before + amount
        assert (contract.functions.atnTotalSupply().call()
                == supply_before + amount)
        assert (contract.functions.balanceOfAt(buyer, mint_block).call()
                == bal_before + amount)
        assert (contract.functions.atnTotalSupplyAt(mint_block).call()
                == supply_before + amount)
        # agentMintTotal (tool-pool earnings) is NOT bumped by a vault mint.
        assert contract.functions.agentMintTotal(buyer).call() == 0

        # A distinct VaultMint event fired.
        mints = contract.events.VaultMint().process_receipt(receipt)
        assert len(mints) == 1
        assert mints[0]["args"]["to"] == buyer
        assert mints[0]["args"]["amount"] == amount

    def _mint_rejected(self, w3, contract, caller: str, to: str,
                       amount: int) -> bool:
        try:
            tx = contract.functions.mintFromVault(to, amount).transact(
                {"from": caller, "gas": 300_000})
            return w3.eth.wait_for_transaction_receipt(tx).status != 1
        except Exception:
            return True

    def test_non_vault_caller_reverts(self):
        abi, bytecode = _load_substrate()
        w3 = Web3(EthereumTesterProvider())
        accounts = w3.eth.accounts
        deployer, vault, intruder, buyer = accounts[:4]
        addr = _deploy(w3, deployer, abi, bytecode, vault_minter=vault)
        contract = w3.eth.contract(address=addr, abi=abi)

        supply_before = contract.functions.atnTotalSupply().call()
        assert self._mint_rejected(w3, contract, intruder, buyer, 1_000)
        assert self._mint_rejected(w3, contract, deployer, buyer, 1_000)
        assert contract.functions.atnTotalSupply().call() == supply_before
        assert contract.functions.balanceOf(buyer).call() == 0
        assert not self._mint_rejected(w3, contract, vault, buyer, 1_000)

    def test_zero_vault_deploy_always_reverts(self):
        abi, bytecode = _load_substrate()
        w3 = Web3(EthereumTesterProvider())
        accounts = w3.eth.accounts
        deployer, buyer = accounts[0], accounts[1]
        addr = _deploy(w3, deployer, abi, bytecode, vault_minter=_ZERO)
        contract = w3.eth.contract(address=addr, abi=abi)

        assert contract.functions.vaultMinter().call() == _ZERO
        for caller in (deployer, buyer):
            assert self._mint_rejected(w3, contract, caller, buyer, 1_000)
        assert contract.functions.atnTotalSupply().call() == 0
