"""RepToken-sourced voice snapshot and the pools' window.

``read_voice_state`` guarantees: voices are priced AS OF the previous
anchor, so activity after it cannot change this epoch's weights, and any
two daemons reading at different times derive the identical maps. The
pools are FEES-ONLY: the window is each index token's tool share collected
between the two cutoffs before the snapshot (lag one), so zero service
volume => an empty window. The value index is read pinned to the same
block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

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
from nodes.common.federated_reconcile import VOICE_EPSILON
from nodes.common.voice_state import read_voice_state
from tests.substrate_chain_helpers import (
    deploy_substrate,
    index_tokens,
    pay_fees,
    register_bound,
    reviewed_close,
    seed_pool,
    tool_registration_batch,
)

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


def _deploy(w3: Web3, deployer: str, abi: list, bytecode: str) -> str:
    # governor = deployer (anchors directly); the RepToken mock below is
    # passed only to read_voice_state.
    return deploy_substrate(w3, abi, bytecode, deployer)


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
    receipt = register_bound(w3, contract, agent_addr, b"peer-" + seed.encode())
    assert receipt.status == 1


_DIGEST = "ef" * 32


def _anchor_epoch(fx, agent_ids, epoch_id: str, rpb: str, window=None):
    """Close + anchor an epoch (agent_ids[0]'s tool's review round concludes
    in it, over ``window``, the pools' new money); returns (close_result,
    resolver)."""
    batches = [tool_registration_batch(rpb, agent_ids[0], _DIGEST)]
    result = reviewed_close(canonical_order(batches), tool_share_window=window)
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


def _claim_for(fx, result, resolver, epoch_id: str):
    """Claim the first earning household's leaf, submitted by a DIFFERENT
    key (the deployer, standing in for a daemon operator), and withdraw it;
    returns (earner address, {token: amount}). The tokens must land on the
    earner, not the sender."""
    w3 = fx["w3"]
    aid = next(a for a in fx["agent_addrs"] if a.lower() in result["claims"])
    sender = fx["deployer"]
    assert sender != aid
    tokens = {t.address.lower(): t for t in index_tokens(fx["contract"])}
    before = {k: t.functions.balanceOf(aid).call() for k, t in tokens.items()}
    sender_before = {k: t.functions.balanceOf(sender).call() for k, t in tokens.items()}
    submitter = AuthoritativeChainSubmitter(
        claimant_id=aid.lower(), claimant_address=aid, resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=fx["addr"]),
        sender_address=sender, web3=w3, substrate_abi=fx["abi"],
    )
    s = submitter.submit_for_epoch(epoch_id)
    assert s.success, s.error
    assert s.tokens and not s.withdraw_errors
    for k, t in tokens.items():
        assert t.functions.balanceOf(aid).call() == before[k] + s.tokens.get(k, 0)
        assert t.functions.balanceOf(sender).call() == sender_before[k]
    return aid, s.tokens


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
    # No seed anchor here: some tests pin the no-anchor (genesis) state.
    # Tests that mint call _seed(fx) first.
    return {
        "w3": w3, "abi": abi, "addr": addr, "contract": contract,
        "deployer": deployer, "agent_addrs": list(agent_addrs),
        "rep_addr": rep_addr, "rep_contract": rep_contract,
    }


def _seed(fx):
    """Pay real fees in every index token and anchor a seed epoch. Returns
    the tool share per token (what the next anchor's cap adds)."""
    return seed_pool(fx["w3"], fx["contract"])


class TestVoiceStateSnapshot:
    def test_no_anchor_means_no_weights(self, fx):
        state = read_voice_state(
            fx["addr"], web3=fx["w3"], rep_token_address=fx["rep_addr"])
        assert state["snapshot_block"] is None
        assert state["voice_weights"] == {}
        assert state["owner_map"] == {}
        # Fees-only: an empty window at genesis.
        assert state["tool_share_window"] == {}
        assert state["parity"] == {}

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

    def test_pinned_read_matches_the_chain_cap(self, fx):
        """Decision 2026-10-26 (pipelined elections): epoch n+1's inputs
        are read at epoch n's CUTOFF. The pinned window is the tool share
        collected between the cutoffs of n-1 and n (what the chain's cap
        adds when the anchor after n lands), and REP is the balance at the
        cutoff, before anything the anchor's own landing time would see."""
        w3, sub = fx["w3"], fx["contract"]
        agents = fx["agent_addrs"]
        _set_votes(w3, fx["rep_contract"], fx["deployer"], agents[0], 1000)
        _seed(fx)                       # epoch 1, cut at its anchoring
        c1 = int(sub.functions.getAnchor(0).call()[10])
        second = _seed(fx)              # epoch 2: collects fees, then cuts
        a2 = sub.functions.getAnchor(1).call()
        c2 = int(a2[10])
        # REP moving after the cutoff does not reach epoch 3's inputs.
        _set_votes(w3, fx["rep_contract"], fx["deployer"], agents[1], 9000)
        state = read_voice_state(
            fx["addr"], web3=w3, rep_token_address=fx["rep_addr"],
            pin={"number": 3, "cutoffs": [c1, c2],
                 "prev_root": bytes(a2[1]).hex()})
        assert state["tool_share_window"] == second and all(v > 0 for v in second.values())
        assert state["snapshot_anchor_count"] == 2
        assert state["rep_shares"][agents[0].lower()] == pytest.approx(1.0)
        assert state["rep_shares"][agents[1].lower()] == 0.0
        win = state["epoch_window"]
        assert win["number"] == 3 and win["start"] == c2
        assert win["prev_root"] == bytes(a2[1]).hex()
        # A pin whose chain head is not past the cutoff is refused.
        future = int(w3.eth.get_block("latest")["timestamp"]) + 100
        with pytest.raises(ValueError):
            read_voice_state(fx["addr"], web3=w3,
                             rep_token_address=fx["rep_addr"],
                             pin={"number": 4, "cutoffs": [c1, c2, future],
                                  "prev_root": "00" * 32})

    def test_delegation_does_not_move_substrate_weight(self, fx):
        """Decision 2026-10-26: the substrate reads REP BALANCES. An agent
        delegating its votes away keeps its household's weight, and votes
        delegated in add none."""
        w3 = fx["w3"]
        agents = fx["agent_addrs"]
        rep = fx["rep_contract"]
        _set_votes(w3, rep, fx["deployer"], agents[0], 1000)
        _set_votes(w3, rep, fx["deployer"], agents[1], 1000)
        # agents[0] delegates everything to agents[1].
        for account, votes in ((agents[0], 0), (agents[1], 2000)):
            tx = rep.functions.setDelegatedVotes(account, votes).transact(
                {"from": fx["deployer"], "gas": 300_000})
            assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        _anchor_epoch(fx, agents, "e_d1", "rpb_d1")
        state = read_voice_state(
            fx["addr"], web3=w3, rep_token_address=fx["rep_addr"])
        assert state["rep_shares"][agents[0].lower()] == pytest.approx(0.5)
        assert state["rep_shares"][agents[1].lower()] == pytest.approx(0.5)
        assert state["agent_rep"][agents[0].lower()] == 1000

    def test_pinned_to_an_earlier_anchor_and_per_agent_rep(self, fx):
        """``anchor_index`` reads as if the chain ended at that anchor (the
        robot poll rail pins each round to one); ``agent_rep`` carries the
        raw REP of every agent at that snapshot."""
        w3 = fx["w3"]
        agents = fx["agent_addrs"]
        _set_votes(w3, fx["rep_contract"], fx["deployer"], agents[0], 1000)
        _anchor_epoch(fx, agents, "e_p1", "rpb_p1")
        _set_votes(w3, fx["rep_contract"], fx["deployer"], agents[1], 3000)
        _anchor_epoch(fx, agents, "e_p2", "rpb_p2")
        first = read_voice_state(fx["addr"], web3=w3, rep_token_address=fx["rep_addr"],
                                 anchor_index=0)
        latest = read_voice_state(fx["addr"], web3=w3, rep_token_address=fx["rep_addr"])
        assert first["snapshot_anchor_count"] == 1
        assert latest["snapshot_anchor_count"] == 2
        assert first["agent_rep"][agents[0].lower()] == 1000
        assert first["agent_rep"][agents[1].lower()] == 0
        assert latest["agent_rep"][agents[1].lower()] == 3000
        assert read_voice_state(fx["addr"], web3=w3, rep_token_address=fx["rep_addr"],
                                anchor_index=-1)["agent_rep"] == {}

    def test_revoked_agent_and_child_drop_out(self, fx):
        """A revoked agent and its descendant (via OwnerBound parent links)
        are absent from the snapshot; unrelated agents stay."""
        w3, contract = fx["w3"], fx["contract"]
        parent, other = fx["agent_addrs"][0], fx["agent_addrs"][1]
        child = w3.eth.accounts[4]
        # Child bound under the (self-owned) parent, signed by the owner.
        r = register_bound(w3, contract, child, b"peer-child",
                           owner=parent, parent=parent)
        assert r.status == 1
        _set_votes(w3, fx["rep_contract"], fx["deployer"], parent, 400)
        _set_votes(w3, fx["rep_contract"], fx["deployer"], child, 300)
        _set_votes(w3, fx["rep_contract"], fx["deployer"], other, 300)
        _anchor_epoch(fx, fx["agent_addrs"], "e_rv0", "rpb_rv0")
        before = read_voice_state(
            fx["addr"], web3=w3, rep_token_address=fx["rep_addr"])
        assert child.lower() in before["owner_map"]
        assert parent.lower() in before["owner_map"]

        # Owner (== parent agent) revokes the parent: subtree goes with it.
        tx = contract.functions.revokeAgent(parent, b"").transact(
            {"from": parent, "gas": 300_000})
        assert w3.eth.wait_for_transaction_receipt(tx).status == 1
        assert not contract.functions.isLive(child).call()
        _anchor_epoch(fx, fx["agent_addrs"], "e_rv1", "rpb_rv1")
        after = read_voice_state(
            fx["addr"], web3=w3, rep_token_address=fx["rep_addr"])
        assert parent.lower() not in after["owner_map"]
        assert child.lower() not in after["owner_map"]
        assert other.lower() in after["owner_map"]
        assert parent.lower() not in after["voice_weights"]
        assert child.lower() not in after["voice_weights"]
        assert other.lower() in after["voice_weights"]

    def test_owner_wallet_not_agent_has_no_rep_term(self, fx):
        """A bare wallet that owns no agent (no OwnerBound) appears in no
        household, whatever REP it holds."""
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

    def test_owner_wallet_own_rep_counts_in_its_household(self, fx):
        """Decision 2026-10-27: the owner wallet claims, so it holds the
        household's REP itself. An owner (not a registered agent) bound
        over an agent gets its OWN REP in that household's share, on top
        of the agent's."""
        w3, contract = fx["w3"], fx["contract"]
        agents = fx["agent_addrs"]
        owner_wallet = w3.eth.accounts[6]     # NOT a registered agent
        bound = w3.eth.accounts[5]
        r = register_bound(w3, contract, bound, b"peer-owned",
                           owner=owner_wallet)
        assert r.status == 1
        _set_votes(w3, fx["rep_contract"], fx["deployer"], owner_wallet, 600)
        _set_votes(w3, fx["rep_contract"], fx["deployer"], bound, 100)
        _set_votes(w3, fx["rep_contract"], fx["deployer"], agents[0], 300)
        _anchor_epoch(fx, agents, "e_ow2", "rpb_ow2")

        state = read_voice_state(
            fx["addr"], web3=w3, rep_token_address=fx["rep_addr"])
        assert state["supply"] == 1000
        assert state["owner_map"][bound.lower()] == owner_wallet.lower()
        house = owner_wallet.lower()
        # 600 (owner's own) + 100 (bound agent) of 1000.
        assert state["rep_shares"][house] == pytest.approx(0.7)
        assert state["voice_weights"][house] == pytest.approx(
            VOICE_EPSILON + 0.7)
        assert state["rep_shares"][agents[0].lower()] == pytest.approx(0.3)
        # The bound agent is not its own household.
        assert bound.lower() not in state["rep_shares"]


class TestToolShareWindow:
    def test_fees_enter_exactly_one_window(self, fx):
        """A service payment's tool share enters the window of the FIRST
        close whose snapshot spans it, then leaves; claims of it are paid in
        the token it was collected in."""
        contract = fx["contract"]
        tokens = index_tokens(contract)
        seed = _seed(fx)                 # anchor 1: its window holds the seed fees
        assert len(seed) == 2 and all(v > 0 for v in seed.values())
        state_1 = read_voice_state(
            fx["addr"], web3=fx["w3"], rep_token_address=fx["rep_addr"])
        assert state_1["tool_share_window"] == seed
        assert state_1["snapshot_anchor_count"] == 1

        # Epoch 2 allocates the seed window; Substrate's cap covers it once
        # anchor 2 lands (poolCutoff = anchor 1's cutoff).
        r1, res1 = _anchor_epoch(fx, fx["agent_addrs"], "e_f1", "rpb_f1",
                                 window=state_1["tool_share_window"])
        for t in tokens:
            assert contract.functions.poolCap(t.address).call() == seed[t.address.lower()]
        _claim_for(fx, r1, res1, "e_f1")
        state_2 = read_voice_state(
            fx["addr"], web3=fx["w3"], rep_token_address=fx["rep_addr"])
        assert state_2["tool_share_window"] == {}     # nothing paid since

        # A payment in one token only: the next window holds it alone.
        paid = pay_fees(fx["w3"], contract, 10 ** 9, tokens=tokens[:1])
        _anchor_epoch(fx, fx["agent_addrs"], "e_f2", "rpb_f2")
        state_3 = read_voice_state(
            fx["addr"], web3=fx["w3"], rep_token_address=fx["rep_addr"])
        assert state_3["tool_share_window"] == paid
        assert state_3["snapshot_anchor_count"] == contract.functions.anchorCount().call()

        # No payments since: the fee has left the window.
        _anchor_epoch(fx, fx["agent_addrs"], "e_f3", "rpb_f3")
        state_4 = read_voice_state(
            fx["addr"], web3=fx["w3"], rep_token_address=fx["rep_addr"])
        assert state_4["tool_share_window"] == {}


class TestParity:
    def test_parity_from_registry_writes(self):
        from nodes.common.voice_state import parity_from_writes
        usdc, dai = "0x" + "a6" * 20, "0x" + "1b" * 20
        assert parity_from_writes({
            "jurisdiction.parity." + usdc: str(10 ** 30),
            "jurisdiction.parity." + dai.upper().replace("0X", "0x"): str(10 ** 18),
            "jurisdiction.parity.0xshort": "5",
            "jurisdiction.parity." + "0x" + "cc" * 20: "",
            "jurisdiction.parity." + "0x" + "dd" * 20: "0",
            "jurisdiction.parity." + "0x" + "ee" * 20: "1.5",
            "jurisdiction.autonet.charter": "{}",
        }) == {dai: 10 ** 18, usdc: 10 ** 30}
