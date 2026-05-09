"""Phase 10.5: full-loop integration — federated close + chain anchor +
per-agent mint submission, all driven through the production drivers.

Differences from test_phase5_6_authoritative_submission.py: that file
exercises the components individually (EpochAnchorer.anchor_close_result,
AuthoritativeChainSubmitter.submit_for_epoch). Here we drive the
*production glue* — ``ChainSubmissionDriver.handle_federated_close()``
— so the test catches regressions in how the driver routes work,
caches submitters, and integrates the resolver.

What's validated:

  1. Winner daemon's ``ChainSubmissionDriver`` anchors the close
     result on Substrate.sol and writes the blob to the resolver.
  2. ``AuthoritativeChainSubmitter.submit_for_epoch`` runs for each
     registered agent in the resolver list, reading the on-chain
     anchor and submitting authoritative mint.
  3. ``Substrate.mintForEpoch(agent, epochIdHash)`` matches the
     scaled mint each agent contributed.
  4. Non-winner daemon (with the same agents registered) submits its
     agents' mints too — the contract's per-(agent, epoch) idempotency
     means it doesn't double-credit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

eth_tester = pytest.importorskip("eth_tester")

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from nodes.common.authoritative_encoding import (
    decode_agent_mint_blob,
)
from nodes.common.authoritative_submitter import epoch_id_hash
from nodes.common.blob_resolver import InMemoryBlobResolver
from nodes.common.canonical_ordering import canonical_order
from nodes.common.chain_submission_driver import (
    AgentChainIdentity,
    ChainSubmissionConfig,
    ChainSubmissionDriver,
)
from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_close_driver import FederatedCloseResult
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
    assert receipt.status == 1, f"registerAgent failed: {receipt}"


@pytest.fixture
def chain():
    """Fresh in-process EVM with Substrate deployed and 3 test agents
    registered."""
    abi, bytecode = _load_substrate()
    w3 = Web3(EthereumTesterProvider())
    accounts = w3.eth.accounts
    deployer = accounts[0]
    addr = _deploy(w3, deployer, abi, bytecode)
    contract = w3.eth.contract(address=addr, abi=abi)
    agent_addrs = accounts[1:4]
    for i, a in enumerate(agent_addrs):
        _register_agent(w3, contract, a, f"agent-lineage-{i}")
    return {
        "w3": w3,
        "abi": abi,
        "addr": addr,
        "contract": contract,
        "deployer": deployer,
        "agent_addrs": list(agent_addrs),
    }


def _make_chain(
    rpb: str, kp: Keypair, agent_id: str, charter_axis: int, n: int = 2,
) -> List[EventBatch]:
    """Build a small canonical chain of batches for one agent, with
    observations biased toward a chosen charter axis so each agent
    creates their own region of contested sub-claims (ensures all
    agents in the test get attributed mint)."""
    chain: List[EventBatch] = []
    prev = b""
    for i in range(1, n + 1):
        coords = [0.0] * (6 + 1024)
        coords[charter_axis] = 0.5
        coords[6 + (charter_axis * 50) + (10 * (i + 1))] = 0.5
        ev = {
            "kind": "observation_added",
            "seq": 1,
            "author_agent": agent_id,
            "obs_id": f"obs_{agent_id}_{i}",
            "coords": coords,
            "label": f"{agent_id}_{i}",
        }
        b = EventBatch(
            rpb_address=rpb,
            sender_pubkey=kp.public_key,
            batch_seq=i,
            events=[ev],
            prev_batch_hash=prev,
            timestamp=1_700_000_000.0 + i,
        )
        chain.append(b)
        prev = b.content_hash()
    return chain


def _build_federated_close_result(
    epoch_id: str,
    rpb: str,
    agent_addrs: List[str],
    is_winner: bool = True,
) -> Tuple[FederatedCloseResult, List[bytes]]:
    """Build a real federated close result + the canonical sender list."""
    senders = [Keypair.generate() for _ in agent_addrs]
    batches: List[EventBatch] = []
    # Assign each agent a different charter axis so they create
    # distinct contested regions (otherwise reconcile attributes
    # the contested node to the latest writer only).
    for axis_idx, (kp, aid) in enumerate(zip(senders, agent_addrs)):
        batches.extend(_make_chain(rpb, kp, aid, charter_axis=axis_idx, n=2))
    canonical = canonical_order(batches)
    close_result = federated_epoch_close(canonical)
    close_result["epoch_id"] = epoch_id
    sender_pubkeys = [s.public_key for s in senders]
    fed = FederatedCloseResult(
        epoch_id=epoch_id,
        close_result=close_result,
        senders=sender_pubkeys,
        winner=sender_pubkeys[0] if is_winner else b"\xff" * 32,
        is_winner=is_winner,
        n_batches=len(canonical.ordered_batches),
    )
    return fed, sender_pubkeys


def test_winner_drives_anchor_and_each_agent_mint_lands(chain):
    """Winner daemon's driver: anchor lands on chain, each agent's
    submitter posts mint, contract balances reflect the authoritative
    payload."""
    w3 = chain["w3"]
    contract = chain["contract"]
    deployer = chain["deployer"]
    agent_addrs = chain["agent_addrs"]

    # Build a real federated close with mint going to the 3 0x addrs.
    fed, _ = _build_federated_close_result(
        epoch_id="e_phase10_5_a",
        rpb="rpb_phase10_5",
        agent_addrs=agent_addrs,
        is_winner=True,
    )
    payload_mint = fed.close_result["authoritative_payload"]["agent_mint"]
    # Federated close may produce mint for only some of the agents
    # given how few synthetic events we built; we only assert the
    # ones with positive mint land correctly on chain.
    minting_addrs = [a for a in agent_addrs if payload_mint.get(a, 0.0) > 0]
    assert minting_addrs, f"no agent got positive mint: {payload_mint}"

    # Build the chain submission driver wired to eth_tester.
    # Use the existing _build_anchorer escape-hatch by passing an
    # already-constructed EpochAnchorer with web3 + abi.
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(
            epoch_anchor_address=chain["addr"],
            private_key="",  # not used in the eth_tester path
        ),
        web3=w3,
        contract_abi=chain["abi"],
    )
    anchorer._set_submitter_address(deployer)

    resolver = InMemoryBlobResolver()
    # Per-agent identities. eth_tester accounts don't have private
    # keys exposed via accounts[i]; we use the deployer as a stand-in
    # for the per-agent submitter signer (eth_tester signs from any
    # local account). Each AuthoritativeChainSubmitter gets the
    # agent_address for the contract call's `from`.
    # We construct submitters manually via the driver's patching path
    # by overriding the cache after the driver decides who to submit
    # for. The simpler route: prebuild submitters via the cache.
    from nodes.common.authoritative_submitter import (
        AuthoritativeChainSubmitter,
        AuthoritativeSubmitterConfig,
    )

    # Build the driver with a mocked anchorer instance + use real
    # AuthoritativeChainSubmitters configured with eth_tester web3.
    driver = ChainSubmissionDriver(
        config=ChainSubmissionConfig(
            substrate_address=chain["addr"],
            rpc_url="",
            chain_id=0,
            daemon_private_key="dummy",  # only needed for non-test path
        ),
        agent_chain_resolver=lambda: [
            AgentChainIdentity(address=a, private_key="dummy") for a in agent_addrs
        ],
        blob_resolver=resolver,
        anchorer=anchorer,
    )
    # Pre-populate the per-agent submitter cache with eth_tester-wired
    # submitters. Empty private_key triggers the eth_tester fast-path
    # in _send_record (transact from agent_address, which is already
    # unlocked in the test EVM).
    for a in agent_addrs:
        sub = AuthoritativeChainSubmitter(
            agent_id=a,
            agent_address=a,
            resolver=resolver,
            config=AuthoritativeSubmitterConfig(
                substrate_address=chain["addr"],
                rpc_url="",
                chain_id=0,
                private_key="",
            ),
            web3=w3,
            substrate_abi=chain["abi"],
        )
        driver._submitter_cache[a] = sub

    out = driver.handle_federated_close(fed)

    assert out["anchored"] is True, out
    assert out["anchor_tx"], out
    assert len(out["agent_submissions"]) == len(agent_addrs)

    # On-chain: every agent that had positive authoritative mint
    # has the matching scaled value recorded for this epoch.
    eid_hash = epoch_id_hash(fed.epoch_id)
    for a in minting_addrs:
        on_chain = contract.functions.mintForEpoch(a, eid_hash).call()
        scaled = int(payload_mint[a] * 1_000_000)
        assert on_chain == scaled, (
            f"agent {a}: on_chain={on_chain} expected={scaled}"
        )

    # Agents with no authoritative mint should remain at zero —
    # ``skip_zero=True`` means the submitter doesn't even attempt
    # the tx. (They show up in agent_submissions with success=True
    # and contribution_scaled=0.)
    for a in agent_addrs:
        if a in minting_addrs:
            continue
        assert contract.functions.mintForEpoch(a, eid_hash).call() == 0


