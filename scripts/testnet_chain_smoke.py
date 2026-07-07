"""Smoke test the chain submission drivers against deployed Substrate.sol.

This is the testnet counterpart to `tests/test_phase10_5_chain_full_loop.py`
(which runs against eth_tester). Same drivers, same flow; the only
change is rpc_url + private keys point at shadownet instead of an
in-process EVM.

What it does:
  1. Reads `deployments/shadownet.json` for the deployed contract address.
  2. Reads `deployments/shadownet-smoke-wallet.json` for the second
     funded wallet.
  3. Registers both wallets as agents on the contract (skipping if
     already registered).
  4. Builds a synthetic 2-sender federated close result.
  5. Calls ChainSubmissionDriver.handle_federated_close using a real
     EpochAnchorer. Verifies anchor and per-agent mint land on chain.

Usage:
  PRIVATE_KEY_A=<funded hex> python scripts/testnet_chain_smoke.py

Where PRIVATE_KEY_A is the deployer/funder wallet (signs the anchor).
The second wallet is read from the smoke-wallet JSON.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if (_REPO_ROOT / "nodes" / "__init__.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))

from web3 import Web3
from eth_account import Account

from nodes.common.authoritative_submitter import (
    AuthoritativeChainSubmitter,
    AuthoritativeSubmitterConfig,
    epoch_id_hash,
)
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


def _load_substrate_abi():
    artifact = json.loads(
        (
            _REPO_ROOT
            / "artifacts"
            / "contracts"
            / "core"
            / "Substrate.sol"
            / "Substrate.json"
        ).read_text()
    )
    return artifact["abi"]


def _make_chain(rpb, kp, agent_id, charter_axis, n, embedding_dim):
    """v3 mint fixture (mint = tool usage only): the agent authors a
    pinned tool that gets a distinct-fleet greenlight and third-party
    attested usage. Observation events alone no longer mint."""
    import hashlib

    digest = hashlib.sha256(f"smoke-tool-{agent_id}".encode()).hexdigest()
    coords = [0.0] * (6 + embedding_dim)
    coords[4] = 0.8
    coords[6 + (charter_axis % embedding_dim)] = 0.5
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

    def _vet(vetter):
        return {
            "kind": "tool_used", "seq": 1, "author_agent": vetter,
            "manifest_digest": digest, "tool_author": agent_id,
            "receipt_digest": hashlib.sha256(
                f"vet-{agent_id}-{vetter}".encode()).hexdigest(),
            "ok": True, "fee_atn": 0.0, "vet": True,
        }

    def _receipt(caller):
        return {
            "kind": "tool_used", "seq": 1, "author_agent": caller,
            "manifest_digest": digest, "tool_author": agent_id,
            "receipt_digest": hashlib.sha256(
                f"use-{agent_id}-{caller}".encode()).hexdigest(),
            "ok": True, "fee_atn": 0.0, "attested": True, "score": 0.8,
        }

    events = [reg, _vet("vetter-1"), _vet("vetter-2"),
              _receipt("caller-1"), _receipt("caller-2")]
    # The registration rides this agent's keypair; every vet/receipt
    # gets its own wire identity (the wire-level self-attestation
    # dedup would otherwise zero the mint).
    chain = [EventBatch(
        rpb_address=rpb, sender_pubkey=kp.public_key, batch_seq=1,
        events=[events[0]], prev_batch_hash=b"", timestamp=1.0,
    )]
    for ev in events[1:]:
        chain.append(EventBatch(
            rpb_address=rpb, sender_pubkey=Keypair.generate().public_key,
            batch_seq=1, events=[ev], prev_batch_hash=b"",
            timestamp=1.0,
        ))
    return chain


def main():
    deploy = json.loads(
        (_REPO_ROOT / "deployments" / "shadownet.json").read_text()
    )
    smoke_wallet = json.loads(
        (_REPO_ROOT / "deployments" / "shadownet-smoke-wallet.json").read_text()
    )

    rpc_url = deploy["rpc_url"]
    chain_id = deploy["chainId"]
    substrate_address = deploy["substrate_address"]

    pk_a = os.environ.get("PRIVATE_KEY_A")
    if not pk_a:
        print("ERROR: set PRIVATE_KEY_A=<hex of funded wallet>", file=sys.stderr)
        sys.exit(1)
    if pk_a.startswith("0x"):
        pk_a = pk_a[2:]
    pk_b = smoke_wallet["target_private_key"]
    if pk_b.startswith("0x"):
        pk_b = pk_b[2:]

    addr_a = Account.from_key("0x" + pk_a).address
    addr_b = Account.from_key("0x" + pk_b).address

    print(f"=== Shadownet chain smoke ===")
    print(f"Substrate: {substrate_address}")
    print(f"Wallet A:  {addr_a}")
    print(f"Wallet B:  {addr_b}")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    abi = _load_substrate_abi()
    contract = w3.eth.contract(address=Web3.to_checksum_address(substrate_address), abi=abi)

    # 1. Register both agents (skip if already registered).
    for label, pk, addr in (("A", pk_a, addr_a), ("B", pk_b, addr_b)):
        already = contract.functions.isRegistered(addr).call()
        if already:
            print(f"[reg] {label} already registered")
            continue
        print(f"[reg] {label} registering...")
        lineage = Web3.keccak(text=f"smoke-lineage-{label}-{int(time.time())}")
        # Synthetic libp2p PeerId placeholder — real daemon registration
        # will pass the actual PeerId bytes from libp2p_host.get_id().
        peer_id = b"smoke-peer-" + label.encode() + b"-" + str(int(time.time())).encode()
        nonce = w3.eth.get_transaction_count(addr)
        tx = contract.functions.registerAgent(lineage, peer_id).build_transaction({
            "from": addr,
            "nonce": nonce,
            "gas": 1_500_000,
            "gasPrice": w3.eth.gas_price,
            "chainId": chain_id,
        })
        signed = w3.eth.account.sign_transaction(tx, "0x" + pk)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        h = w3.eth.send_raw_transaction(raw)
        receipt = w3.eth.wait_for_transaction_receipt(h)
        if receipt.status != 1:
            raise RuntimeError(f"registerAgent({label}) reverted: {receipt}")
        print(f"[reg] {label} ok: tx={h.hex()}")

    # 2. Build a synthetic federated close result with mint to both addrs.
    epoch_id = f"shadownet-smoke-{int(time.time())}"
    rpb = "rpb_shadownet_smoke"
    senders = [Keypair.generate() for _ in range(2)]
    agent_ids = [addr_a, addr_b]
    embedding_dim = 16
    batches = []
    for axis_idx, (kp, aid) in enumerate(zip(senders, agent_ids)):
        batches.extend(
            _make_chain(rpb, kp, aid, charter_axis=axis_idx, n=2, embedding_dim=embedding_dim)
        )
    canonical = canonical_order(batches)
    print(f"[close] canonical: {len(canonical.ordered_batches)} batches, "
          f"{len(canonical.per_sender_root)} senders")
    close_result = federated_epoch_close(canonical, embedding_dim=embedding_dim)
    close_result["epoch_id"] = epoch_id
    payload_mint = close_result["authoritative_payload"]["agent_mint"]
    print(f"[close] payload agent_mint: {payload_mint}")
    minting_addrs = [a for a in agent_ids if payload_mint.get(a, 0.0) > 0]
    if not minting_addrs:
        print("ERROR: no agent got positive authoritative mint", file=sys.stderr)
        sys.exit(1)
    print(f"[close] minting addrs: {minting_addrs}")

    # 3. Construct FederatedCloseResult (the driver's input shape).
    fed = FederatedCloseResult(
        epoch_id=epoch_id,
        close_result=close_result,
        senders=[s.public_key for s in senders],
        winner=senders[0].public_key,
        is_winner=True,
        n_batches=len(canonical.ordered_batches),
    )

    # 4. Build chain submission driver, prepopulate per-agent submitters.
    config = ChainSubmissionConfig(
        substrate_address=substrate_address,
        rpc_url=rpc_url,
        chain_id=chain_id,
        daemon_private_key=pk_a,  # Wallet A signs the anchor
    )
    resolver = InMemoryBlobResolver()
    driver = ChainSubmissionDriver(
        config=config,
        agent_chain_resolver=lambda: [
            AgentChainIdentity(address=addr_a, private_key=pk_a),
            AgentChainIdentity(address=addr_b, private_key=pk_b),
        ],
        blob_resolver=resolver,
    )
    print(f"[driver] enabled={driver.enabled}")
    if not driver.enabled:
        print("ERROR: driver disabled", file=sys.stderr)
        sys.exit(1)

    # 5. Drive the close → anchor + per-agent submission.
    print(f"[driver] handle_federated_close...")
    out = driver.handle_federated_close(fed)
    print(f"[driver] anchored={out['anchored']} tx={out['anchor_tx']}")
    if not out["anchored"]:
        print(f"ERROR: anchor failed: {out.get('anchor_error')}", file=sys.stderr)
        sys.exit(1)
    for sub in out["agent_submissions"]:
        print(f"[driver] {sub['agent_address']}: success={sub['success']} "
              f"raw_mint={sub['raw_mint']} tx={sub['tx_hash']} "
              f"err={sub.get('error', '')}")

    # 6. Read on-chain state and confirm.
    eid_hash = epoch_id_hash(epoch_id)
    print(f"\n[verify] epoch_id_hash: 0x{eid_hash.hex()}")
    print(f"[verify] isAnchored: {contract.functions.isAnchored(epoch_id).call()}")
    for addr in agent_ids:
        on_chain = contract.functions.mintForEpoch(addr, eid_hash).call()
        expected_raw = payload_mint.get(addr, 0.0)
        expected_scaled = int(expected_raw * 1_000_000)
        match = "OK" if on_chain == expected_scaled else "MISMATCH"
        print(f"[verify] {addr}: on_chain={on_chain} expected={expected_scaled} {match}")
    total_a = contract.functions.agentMintTotal(addr_a).call()
    total_b = contract.functions.agentMintTotal(addr_b).call()
    print(f"[verify] agentMintTotal A: {total_a}")
    print(f"[verify] agentMintTotal B: {total_b}")
    print("DONE")


if __name__ == "__main__":
    main()
