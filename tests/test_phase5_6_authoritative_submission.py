"""Phase 5.6: agents read authoritative mint from anchor blob and
submit on-chain via Substrate.recordTrainingForEpoch.

This test file is the **architectural successor** to
``test_phase5_1_chain_submission.py``. The flip:

  - 5.1 encoded "agent submits the projection mint computed by the
    local daemon's close." Best-effort, projection-level.
  - 5.6 encodes "agent reads the consensus-ratified authoritative
    mint from the on-chain anchor blob and submits that exact value."
    Federation-ratified, contract-level idempotent.

What this validates:

  1. End-to-end: WorldService close → canonical_order →
     federated_epoch_close → anchor on Substrate → agent reads anchor
     → fetches blob via resolver → CID-verifies → decodes own share
     → submits on chain.
  2. ``Substrate.mintForEpoch(agent, epochIdHash)`` matches the scaled
     authoritative mint.
  3. Per-(agent, epoch_id) idempotency is enforced by the contract:
     a second submission for the same (agent, epoch_id) reverts with
     ``AlreadySubmittedForEpoch``, and the submitter degrades to a
     no-op success.
  4. Process restart (fresh submitter, no in-memory seen-set) does
     NOT double-count: the contract's idempotency catches it.
  5. Three-agent federation: each agent reads its own attribution
     from the same blob, all three submit independently, on-chain
     totals match the authoritative_payload.
  6. Blob integrity: a tampered blob from the resolver is rejected.
  7. Zero-mint agents skip the tx (gas savings, configurable).
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

from nodes.common.authoritative_encoding import (
    cid_for_blob,
    decode_agent_mint_blob,
    encode_agent_mint_blob,
)
from nodes.common.authoritative_submitter import (
    AuthoritativeChainSubmitter,
    AuthoritativeSubmitterConfig,
    epoch_id_hash,
)
from nodes.common.blob_resolver import (
    BlobIntegrityError,
    InMemoryBlobResolver,
)
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
    tx = contract.constructor(deployer, "0x0000000000000000000000000000000000000000").transact({"from": deployer, "gas": 8_000_000})
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
    """Fresh in-process EVM with Substrate deployed and three test
    agents registered."""
    abi, bytecode = _load_substrate()
    w3 = Web3(EthereumTesterProvider())
    accounts = w3.eth.accounts
    deployer = accounts[0]
    addr = _deploy(w3, deployer, abi, bytecode)
    contract = w3.eth.contract(address=addr, abi=abi)

    agent_addrs = accounts[1:4]
    for i, addr_ in enumerate(agent_addrs):
        _register_agent(w3, contract, addr_, f"agent-lineage-{i}")

    return {
        "w3": w3,
        "abi": abi,
        "addr": addr,
        "contract": contract,
        "deployer": deployer,
        "agent_addrs": list(agent_addrs),
    }


# ---------------------------------------------------------------------------
# Helpers to build a federated close
# ---------------------------------------------------------------------------


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


def _build_close_and_anchor(
    chain_fixture: Dict[str, Any],
    agent_ids: List[str],
    epoch_id: str = "e_5_6",
    rpb: str = "rpb_5_6",
) -> Tuple[Dict[str, Any], str, InMemoryBlobResolver]:
    """Build a federated close from agents, anchor it on chain, and
    populate a resolver with the blob."""
    w3 = chain_fixture["w3"]
    deployer = chain_fixture["deployer"]

    # Build chains for each agent (each gets its own keypair).
    senders = [Keypair.generate() for _ in agent_ids]
    batches: List[EventBatch] = []
    for kp, aid in zip(senders, agent_ids):
        batches.extend(_make_chain(rpb, kp, aid, n=2))

    canonical = canonical_order(batches)
    result = federated_epoch_close(canonical)
    result["epoch_id"] = epoch_id

    # Anchor on chain.
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain_fixture["addr"]),
        web3=w3,
        contract_abi=chain_fixture["abi"],
    )
    anchor_result = anchorer.anchor_close_result(result, from_address=deployer)
    assert anchor_result.success, anchor_result.error

    # Populate the resolver with the blob.
    resolver = InMemoryBlobResolver()
    cid = resolver.put(anchor_result.agent_mint_blob)
    assert cid == anchor_result.agent_mint_cid, "CID mismatch"

    return result, epoch_id, resolver


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def test_agent_reads_anchor_and_submits_authoritative_mint(chain):
    """End-to-end: build close → anchor → agent reads → submits.
    on-chain mintForEpoch matches the authoritative blob."""
    agent_addrs = chain["agent_addrs"]
    contract = chain["contract"]
    # Agent ids ARE addresses: the merkle mint proof is keyed by
    # msg.sender, so claimable mint-map entries must be address-keyed
    # (matches ChainSubmissionDriver, which uses agent_id=address).
    agent_ids = list(agent_addrs)

    result, epoch_id, resolver = _build_close_and_anchor(chain, agent_ids)
    expected_blob_mint = decode_agent_mint_blob(resolver.get(
        result["authoritative_payload"]["agent_mint"]
        and cid_for_blob(encode_agent_mint_blob(result["agent_mint"]))
    ))

    # Pick the first agent that actually has mint; submit for it.
    minting_agent_id = next((aid for aid in agent_ids if expected_blob_mint.get(aid, 0) > 0), None)
    if minting_agent_id is None:
        pytest.skip("federated close produced no positive mint for any test agent")

    minting_addr = agent_addrs[agent_ids.index(minting_agent_id)]

    submitter = AuthoritativeChainSubmitter(
        agent_id=minting_agent_id,
        agent_address=minting_addr,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"],
        substrate_abi=chain["abi"],
    )

    # Precondition: nothing submitted yet for this (agent, epoch).
    eid_hash = epoch_id_hash(epoch_id)
    assert contract.functions.mintForEpoch(minting_addr, eid_hash).call() == 0

    submission = submitter.submit_for_epoch(epoch_id)
    assert submission.success, submission.error
    assert submission.cid_verified is True
    assert submission.contribution_scaled > 0
    expected_scaled = int(expected_blob_mint[minting_agent_id] * 1_000_000)
    assert submission.contribution_scaled == expected_scaled

    # On-chain state matches the authoritative scaled mint.
    on_chain = contract.functions.mintForEpoch(minting_addr, eid_hash).call()
    assert on_chain == expected_scaled, (on_chain, expected_scaled)
    assert contract.functions.agentMintTotal(minting_addr).call() == expected_scaled


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_contract_dedupes_double_submit(chain):
    """Submit twice for the same (agent, epoch). Second call no-ops
    via the contract's AlreadySubmittedForEpoch revert."""
    agent_addrs = chain["agent_addrs"]
    contract = chain["contract"]
    agent_ids = list(agent_addrs[:2])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_idem",
    )
    minting_agent_id = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid, 0) > 0), None,
    )
    if minting_agent_id is None:
        pytest.skip("no positive mint for idempotency test")
    minting_addr = agent_addrs[agent_ids.index(minting_agent_id)]

    submitter = AuthoritativeChainSubmitter(
        agent_id=minting_agent_id,
        agent_address=minting_addr,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"],
        substrate_abi=chain["abi"],
    )
    s1 = submitter.submit_for_epoch(epoch_id)
    assert s1.success
    on_chain_after_first = contract.functions.agentMintTotal(minting_addr).call()

    s2 = submitter.submit_for_epoch(epoch_id)
    # Second call returns success no-op (in-memory seen-set fast path).
    assert s2.success
    assert s2.contribution_scaled == 0
    on_chain_after_second = contract.functions.agentMintTotal(minting_addr).call()
    assert on_chain_after_first == on_chain_after_second


def test_restart_does_not_double_submit(chain):
    """Process restart: a fresh submitter with empty seen-set tries
    to re-submit. The contract reverts; the submitter degrades to
    success no-op rather than failing loudly. On-chain total
    unchanged."""
    agent_addrs = chain["agent_addrs"]
    contract = chain["contract"]
    agent_ids = list(agent_addrs[:2])
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_restart",
    )
    minting_agent_id = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid, 0) > 0), None,
    )
    if minting_agent_id is None:
        pytest.skip("no positive mint for restart test")
    minting_addr = agent_addrs[agent_ids.index(minting_agent_id)]

    submitter1 = AuthoritativeChainSubmitter(
        agent_id=minting_agent_id, agent_address=minting_addr,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"], substrate_abi=chain["abi"],
    )
    s1 = submitter1.submit_for_epoch(epoch_id)
    assert s1.success and s1.contribution_scaled > 0
    on_chain_total = contract.functions.agentMintTotal(minting_addr).call()

    # Pretend the process restarted: brand-new submitter, no memory.
    submitter2 = AuthoritativeChainSubmitter(
        agent_id=minting_agent_id, agent_address=minting_addr,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"], substrate_abi=chain["abi"],
    )
    s2 = submitter2.submit_for_epoch(epoch_id)
    # Contract refused the duplicate; submitter degrades to no-op
    # success. On-chain total unchanged.
    assert s2.success
    assert contract.functions.agentMintTotal(minting_addr).call() == on_chain_total


# ---------------------------------------------------------------------------
# Three-agent federation
# ---------------------------------------------------------------------------


def test_three_agents_each_submit_independently(chain):
    """Three agents in the same federated close. Each reads its own
    share, each submits, on-chain totals match the authoritative
    payload."""
    agent_addrs = chain["agent_addrs"]
    contract = chain["contract"]
    agent_ids = list(agent_addrs)
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_three",
    )
    eid_hash = epoch_id_hash(epoch_id)

    submissions = {}
    for aid, addr_ in zip(agent_ids, agent_addrs):
        sub = AuthoritativeChainSubmitter(
            agent_id=aid, agent_address=addr_,
            resolver=resolver,
            config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
            web3=chain["w3"], substrate_abi=chain["abi"],
        )
        submissions[aid] = sub.submit_for_epoch(epoch_id)

    expected = result["agent_mint"]
    network_total_expected = 0
    for aid, addr_ in zip(agent_ids, agent_addrs):
        on_chain = contract.functions.mintForEpoch(addr_, eid_hash).call()
        if aid in expected and expected[aid] > 0:
            scaled = int(expected[aid] * 1_000_000)
            assert on_chain == scaled, (aid, on_chain, scaled)
            network_total_expected += scaled
        else:
            # Zero mint: submitter skipped the tx entirely.
            assert on_chain == 0

    assert contract.functions.networkMintTotal().call() == network_total_expected


# ---------------------------------------------------------------------------
# Blob integrity
# ---------------------------------------------------------------------------


def test_tampered_blob_raises_integrity_error(chain):
    """A resolver returning bytes whose CID doesn't match the
    advertised CID indicates tampering. The submitter raises."""
    agent_addrs = chain["agent_addrs"]
    agent_ids = ["alice", "bob"]
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_tamper",
    )
    minting_agent_id = next(
        (aid for aid in agent_ids if result["agent_mint"].get(aid, 0) > 0), None,
    )
    if minting_agent_id is None:
        pytest.skip("no positive mint for tamper test")
    minting_addr = agent_addrs[agent_ids.index(minting_agent_id)]

    # Replace the blob in the resolver with a tampered version that
    # doesn't match its CID.
    real_cid = list(resolver._store.keys())[0]
    resolver._store[real_cid] = b'{"schema":1,"agent_mint":{"alice":"99999999.9999999999"}}'

    submitter = AuthoritativeChainSubmitter(
        agent_id=minting_agent_id, agent_address=minting_addr,
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"], substrate_abi=chain["abi"],
    )
    with pytest.raises(BlobIntegrityError):
        submitter.submit_for_epoch(epoch_id)


def test_missing_blob_returns_clean_failure(chain):
    """Anchor exists on chain but the resolver doesn't have the blob.
    Returns a graceful failure, not a crash."""
    agent_addrs = chain["agent_addrs"]
    agent_ids = ["alice"]
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_missing_blob",
    )
    # Empty the resolver.
    resolver._store.clear()

    submitter = AuthoritativeChainSubmitter(
        agent_id="alice", agent_address=agent_addrs[0],
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"], substrate_abi=chain["abi"],
    )
    submission = submitter.submit_for_epoch(epoch_id)
    assert not submission.success
    assert "blob not found" in submission.error.lower()


# ---------------------------------------------------------------------------
# Zero mint
# ---------------------------------------------------------------------------


def test_agent_with_zero_mint_skips_chain_call(chain):
    """An agent who has no mint share for the epoch shouldn't waste
    gas on a no-op tx."""
    agent_addrs = chain["agent_addrs"]
    contract = chain["contract"]
    agent_ids = ["alice", "bob"]
    result, epoch_id, resolver = _build_close_and_anchor(
        chain, agent_ids, epoch_id="e_zero",
    )
    # Use a non-minting agent_id (charlie isn't in the close)
    submitter = AuthoritativeChainSubmitter(
        agent_id="charlie",
        agent_address=agent_addrs[2],
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"], substrate_abi=chain["abi"],
    )
    submission = submitter.submit_for_epoch(epoch_id)
    assert submission.success
    assert submission.contribution_scaled == 0
    assert submission.tx_hash == ""
    # On-chain: nothing recorded for charlie.
    eid_hash = epoch_id_hash(epoch_id)
    assert contract.functions.mintForEpoch(agent_addrs[2], eid_hash).call() == 0


# ---------------------------------------------------------------------------
# Anchor-binding: contract refuses recordTraining for un-anchored epochs
# ---------------------------------------------------------------------------


def test_unanchored_epoch_rejected_by_contract(chain):
    """Submit for an epoch that hasn't been anchored yet. Contract
    reverts EpochNotAnchored; submitter degrades to graceful failure
    OR success-no-op (the existing 'already submitted on chain' path
    swallows revert reasons; an EpochNotAnchored revert lands in the
    same path)."""
    agent_addrs = chain["agent_addrs"]
    resolver = InMemoryBlobResolver()
    # Pre-populate a fake blob the submitter would otherwise read,
    # so we get past the resolver lookup. But we never anchor the
    # epoch on chain.
    fake_blob = b'{"schema":1,"agent_mint":{"alice":"1.0000000000"}}'
    fake_cid = resolver.put(fake_blob)

    submitter = AuthoritativeChainSubmitter(
        agent_id="alice", agent_address=agent_addrs[0],
        resolver=resolver,
        config=AuthoritativeSubmitterConfig(substrate_address=chain["addr"]),
        web3=chain["w3"], substrate_abi=chain["abi"],
    )

    # The submitter calls getAnchorByEpochId first, which reverts
    # for an unanchored epoch — surfaces as a clean failure.
    submission = submitter.submit_for_epoch("e_never_anchored")
    assert not submission.success
    assert submission.error
