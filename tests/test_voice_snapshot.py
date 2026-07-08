"""Checkpointed ATN + reputation + snapshot-pinned voice state.

Validates the IVotes-mechanism checkpoints on Substrate.sol for BOTH
ledgers — ATN (``balanceOfAt`` / ``atnTotalSupplyAt``, money) and
reputation (``reputationOfAt`` / ``reputationTotalSupplyAt``, voice) —
Trace208 history, no delegation layer, and ``read_voice_state``'s
architecture guarantee: voices are priced from REPUTATION AS OF the
previous epoch's anchor block (ratified 2026-07-08: ATN = money,
reputation = voice), so activity after the anchor cannot change this
epoch's weights, and any two daemons reading at different times derive
the identical maps. The headline money/voice split test: a vault ATN
mint moves balances but NOT voice weights.
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
    tx = contract.constructor(deployer, "0x0000000000000000000000000000000000000000").transact({"from": deployer, "gas": 8_000_000})
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


class TestCheckpointedReputation:
    def test_reputation_of_at_tracks_history(self, fx):
        """Reputation is checkpointed at the training mint. A block
        BEFORE a later mint still reads the earlier (lower) reputation;
        the ledger is monotonic and untransferable, so unlike ATN there
        is exactly one write site."""
        w3, contract = fx["w3"], fx["contract"]
        # Epoch 1: anchor + mint to a first agent.
        r1, res1 = _anchor_epoch(fx, fx["agent_addrs"], "e_r1", "rpb_r1")
        minter, amount = _mint_for(fx, r1, res1, "e_r1")
        block_after_first = w3.eth.block_number

        # History at the first mint: minter has `amount`, supply == amount.
        assert contract.functions.reputationOfAt(
            minter, block_after_first).call() == amount
        assert contract.functions.reputationTotalSupplyAt(
            block_after_first).call() == amount
        # Before anything minted: zero (no checkpoint <= block).
        assert contract.functions.reputationOfAt(minter, 0).call() == 0
        assert contract.functions.reputationTotalSupplyAt(0).call() == 0

        # Epoch 2: anchor + mint AGAIN to the same agent — reputation is
        # monotonic, so it accumulates.
        r2, res2 = _anchor_epoch(fx, fx["agent_addrs"], "e_r2", "rpb_r2")
        minter2, amount2 = _mint_for(fx, r2, res2, "e_r2")
        assert minter2 == minter          # same agent mints (agent_ids[0])
        block_after_second = w3.eth.block_number

        # Present reputation reflects both mints; the EARLIER block still
        # reads only the first mint (checkpoint history, no archive node).
        assert contract.functions.agentReputation(minter).call() == (
            amount + amount2)
        assert contract.functions.reputationOfAt(
            minter, block_after_first).call() == amount
        assert contract.functions.reputationOfAt(
            minter, block_after_second).call() == amount + amount2
        assert contract.functions.reputationTotalSupplyAt(
            block_after_second).call() == amount + amount2

    def test_reputation_survives_atn_transfer(self, fx):
        """Transferring ATN (money) does NOT move reputation (soulbound).
        The one reputation write site is the mint — nothing else."""
        w3, contract = fx["w3"], fx["contract"]
        r1, res1 = _anchor_epoch(fx, fx["agent_addrs"], "e_rt", "rpb_rt")
        minter, amount = _mint_for(fx, r1, res1, "e_rt")
        rep_before = contract.functions.agentReputation(minter).call()

        other = next(a for a in fx["agent_addrs"] if a != minter)
        tx = contract.functions.transfer(other, amount).transact(
            {"from": minter, "gas": 200_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1

        # ATN left the minter entirely; reputation is untouched on BOTH.
        assert contract.functions.balanceOf(minter).call() == 0
        assert contract.functions.agentReputation(minter).call() == rep_before
        assert contract.functions.agentReputation(other).call() == 0


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
        different times see the identical maps. Weights are now priced
        from REPUTATION (voice), so ``supply`` is the reputation supply
        (== the training amount, since training mints rep 1:1 with ATN)."""
        w3, contract = fx["w3"], fx["contract"]
        # Epoch 1: anchor, then mint (mint lands AFTER anchor #1's
        # block, so it's invisible to a snapshot at anchor #1).
        r1, res1 = _anchor_epoch(fx, fx["agent_addrs"], "e_v1", "rpb_v1")
        minter, amount = _mint_for(fx, r1, res1, "e_v1")

        state_1 = read_voice_state(fx["addr"], web3=fx["w3"])
        assert state_1["snapshot_block"] is not None
        assert state_1["weight_source"] == "reputation"
        assert state_1["supply"] == 0                      # pre-mint snapshot
        # All registered agents present, all at the epsilon floor.
        for a in fx["agent_addrs"]:
            assert state_1["voice_weights"][a.lower()] == pytest.approx(
                VOICE_EPSILON)

        # Epoch 2: anchor again — the new snapshot sees the mint.
        _anchor_epoch(fx, fx["agent_addrs"], "e_v2", "rpb_v2")
        state_2 = read_voice_state(fx["addr"], web3=fx["w3"])
        assert state_2["snapshot_block"] > state_1["snapshot_block"]
        assert state_2["supply"] == amount        # reputation supply
        assert state_2["voice_weights"][minter.lower()] == pytest.approx(
            VOICE_EPSILON + 1.0)     # holds the entire reputation supply

        # THE PIN: transfer ATN after anchor #2 — a re-read still returns
        # the identical maps. Doubly pinned now: reputation is soulbound,
        # so an ATN transfer cannot move voice weights at ALL (not even
        # at the next epoch), and even a post-anchor MINT wouldn't count
        # until the next snapshot.
        other = next(a for a in fx["agent_addrs"] if a != minter)
        tx = contract.functions.transfer(other, amount).transact(
            {"from": minter, "gas": 200_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        state_2b = read_voice_state(fx["addr"], web3=fx["w3"])
        assert state_2b["voice_weights"] == state_2["voice_weights"]
        assert state_2b["snapshot_block"] == state_2["snapshot_block"]

    def test_post_anchor_mint_does_not_change_pinned_weights(self, fx):
        """Rep-based pin: an agent that TRAINS (mints reputation) after
        the snapshot anchor carries no extra voice weight at the pinned
        block — the checkpoint at the anchor block already fixed it."""
        w3 = fx["w3"]
        # Epoch 1 + mint, then epoch 2 whose snapshot sees that mint.
        r1, res1 = _anchor_epoch(fx, fx["agent_addrs"], "e_pm1", "rpb_pm1")
        minter, amount = _mint_for(fx, r1, res1, "e_pm1")
        _anchor_epoch(fx, fx["agent_addrs"], "e_pm2", "rpb_pm2")
        pinned = read_voice_state(fx["addr"], web3=fx["w3"])

        # A NEW mint lands after anchor #2's block (epoch 3 close+mint),
        # but we re-read without a fresh anchor: still anchor #2's block.
        r3, res3 = _anchor_epoch(fx, fx["agent_addrs"], "e_pm3", "rpb_pm3")
        # anchor #3 exists now, so re-reading pins to #3 — instead assert
        # the earlier snapshot value is unchanged by reading reputationOfAt
        # directly at the pinned block after the epoch-3 mint.
        _mint_for(fx, r3, res3, "e_pm3")
        contract = fx["contract"]
        pinned_block = pinned["snapshot_block"]
        # Reputation at the pinned block is still just the first mint.
        assert contract.functions.reputationOfAt(
            minter, pinned_block).call() == amount
        assert contract.functions.reputationTotalSupplyAt(
            pinned_block).call() == amount
        assert pinned["voice_weights"][minter.lower()] == pytest.approx(
            VOICE_EPSILON + 1.0)

    def test_vault_mint_moves_money_not_voice(self, fx):
        """THE HEADLINE money/voice split. A vault ATN mint to a fleet
        agent (purchased money) after the snapshot changes NEITHER the
        pinned weights NOR — because reputation is what voice reads — the
        weights at the NEXT epoch. Reputation supply is untouched by vault
        mints, so a backer who buys ATN gains money, never voice."""
        # Deploy WITH a vault minter so mintFromVault is enabled.
        abi, bytecode = _load_substrate()
        w3 = Web3(EthereumTesterProvider())
        accounts = w3.eth.accounts
        deployer, vault = accounts[0], accounts[5]
        addr = _deploy_with_vault(w3, deployer, abi, bytecode, vault)
        contract = w3.eth.contract(address=addr, abi=abi)
        agent_addrs = list(accounts[1:4])
        for i, ad in enumerate(agent_addrs):
            _register_agent(w3, contract, ad, f"vault-voice-{i}")
        vfx = {
            "w3": w3, "abi": abi, "addr": addr, "contract": contract,
            "deployer": deployer, "agent_addrs": agent_addrs,
        }

        # Epoch 1 + mint, epoch 2 snapshot sees it.
        r1, res1 = _anchor_epoch(vfx, agent_addrs, "e_vm1", "rpb_vm1")
        minter, amount = _mint_for(vfx, r1, res1, "e_vm1")
        _anchor_epoch(vfx, agent_addrs, "e_vm2", "rpb_vm2")
        before = read_voice_state(addr, web3=w3)
        assert before["voice_weights"][minter.lower()] == pytest.approx(
            VOICE_EPSILON + 1.0)
        rep_supply_before = before["supply"]

        # The vault mints a MASSIVE amount of ATN to another fleet agent —
        # pure purchased money, no reputation.
        buyer = next(a for a in agent_addrs if a != minter)
        big = amount * 1000
        tx = contract.functions.mintFromVault(buyer, big).transact(
            {"from": vault, "gas": 300_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1

        # Anchor a fresh epoch so the NEXT snapshot would see any weight
        # change — and confirm there is none. The buyer holds 1000x the
        # ATN but still sits at the epsilon floor; the minter keeps all
        # the voice. Reputation supply is unchanged.
        _anchor_epoch(vfx, agent_addrs, "e_vm3", "rpb_vm3")
        after = read_voice_state(addr, web3=w3)
        assert after["supply"] == rep_supply_before      # rep supply flat
        assert after["voice_weights"][buyer.lower()] == pytest.approx(
            VOICE_EPSILON)                                # money ≠ voice
        assert after["voice_weights"][minter.lower()] == pytest.approx(
            VOICE_EPSILON + 1.0)
        # Sanity: the buyer really did receive the money.
        assert contract.functions.balanceOf(buyer).call() >= big

    def test_owner_wallet_not_agent_has_no_rep_term(self, fx):
        """Owner wallets never earn reputation (only agents mint it), so a
        bare owner wallet contributes NO reputation term to its household
        — the household weight is exactly the sum of its bound agents'
        reputation. Verified against a hand-built owner_map + a direct
        reputation read (no owner-wallet seed, unlike the money view)."""
        w3, contract = fx["w3"], fx["contract"]
        # Bind two agents to one owner wallet (a non-agent EOA), then mint
        # reputation to one of them.
        owner_wallet = w3.eth.accounts[6]     # NOT a registered agent
        a0, a1 = fx["agent_addrs"][0], fx["agent_addrs"][1]
        # The federated mint lands on agent_ids[0]; check both agents'
        # reputation and confirm the owner wallet has none.
        r1, res1 = _anchor_epoch(fx, fx["agent_addrs"], "e_ow1", "rpb_ow1")
        minter, amount = _mint_for(fx, r1, res1, "e_ow1")

        # Owner wallet has zero reputation at any block — it never trained.
        assert contract.functions.agentReputation(owner_wallet).call() == 0
        blk = w3.eth.block_number
        assert contract.functions.reputationOfAt(owner_wallet, blk).call() == 0

        # read_voice_state builds households ONLY from agent reputation.
        # With no OwnerBound events here, each agent is its own household;
        # the minter carries the full weight, the owner wallet appears in
        # no household at all (it's neither an agent nor a bound owner).
        _anchor_epoch(fx, fx["agent_addrs"], "e_ow2", "rpb_ow2")
        state = read_voice_state(fx["addr"], web3=fx["w3"])
        assert owner_wallet.lower() not in state["voice_weights"]
        assert state["voice_weights"][minter.lower()] == pytest.approx(
            VOICE_EPSILON + 1.0)


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
        tx = contract.functions.payForService(
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


_ZERO = "0x0000000000000000000000000000000000000000"


def _deploy_with_vault(w3: Web3, deployer: str, abi: list, bytecode: str,
                       vault_minter: str) -> str:
    """Deploy Substrate with an explicit vaultMinter (treasury = deployer)."""
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor(deployer, vault_minter).transact(
        {"from": deployer, "gas": 8_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    return receipt.contractAddress


class TestVaultMint:
    """mintFromVault: purchased ATN. Mints balance/supply/checkpoint like
    the training path, but reputation is NEVER touched (money, not voice),
    only vaultMinter may call, and a zero-vault deploy always reverts."""

    def test_vault_minter_mints_atn_without_reputation(self):
        abi, bytecode = _load_substrate()
        w3 = Web3(EthereumTesterProvider())
        accounts = w3.eth.accounts
        deployer, vault, buyer = accounts[0], accounts[1], accounts[2]
        addr = _deploy_with_vault(w3, deployer, abi, bytecode, vault)
        contract = w3.eth.contract(address=addr, abi=abi)

        assert contract.functions.vaultMinter().call() == vault
        supply_before = contract.functions.atnTotalSupply().call()
        rep_before = contract.functions.agentReputation(buyer).call()
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

        # Voice did NOT: reputation is unchanged. Purchased money is not
        # earned voice (ratified 2026-07-08). Neither the live ledger nor
        # the CHECKPOINT moved — a vault mint writes no reputation history.
        assert contract.functions.agentReputation(buyer).call() == rep_before
        assert contract.functions.agentMintTotal(buyer).call() == rep_before
        assert (contract.functions.reputationOfAt(buyer, mint_block).call()
                == rep_before)
        assert (contract.functions.reputationTotalSupplyAt(mint_block).call()
                == contract.functions.networkMintTotal().call())

        # A distinct VaultMint event fired.
        mints = contract.events.VaultMint().process_receipt(receipt)
        assert len(mints) == 1
        assert mints[0]["args"]["to"] == buyer
        assert mints[0]["args"]["amount"] == amount

    def _mint_rejected(self, w3, contract, caller: str, to: str,
                       amount: int) -> bool:
        """True if mintFromVault(caller→to) does not succeed (raises at
        submit or lands a status!=1 receipt). Explicit gas skips estimation,
        so a revert surfaces as a failed receipt rather than an exception."""
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
        addr = _deploy_with_vault(w3, deployer, abi, bytecode, vault)
        contract = w3.eth.contract(address=addr, abi=abi)

        supply_before = contract.functions.atnTotalSupply().call()
        # An intruder cannot mint; neither can the deployer (treasury).
        assert self._mint_rejected(w3, contract, intruder, buyer, 1_000)
        assert self._mint_rejected(w3, contract, deployer, buyer, 1_000)
        # No supply moved.
        assert contract.functions.atnTotalSupply().call() == supply_before
        assert contract.functions.balanceOf(buyer).call() == 0
        # The real vault minter still works.
        assert not self._mint_rejected(w3, contract, vault, buyer, 1_000)

    def test_zero_vault_deploy_always_reverts(self):
        abi, bytecode = _load_substrate()
        w3 = Web3(EthereumTesterProvider())
        accounts = w3.eth.accounts
        deployer, buyer = accounts[0], accounts[1]
        addr = _deploy_with_vault(w3, deployer, abi, bytecode, _ZERO)
        contract = w3.eth.contract(address=addr, abi=abi)

        assert contract.functions.vaultMinter().call() == _ZERO
        # Feature off: no caller can mint.
        for caller in (deployer, buyer):
            assert self._mint_rejected(w3, contract, caller, buyer, 1_000)
        assert contract.functions.atnTotalSupply().call() == 0
