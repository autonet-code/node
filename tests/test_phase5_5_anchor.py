"""Phase 5.5: world anchoring on chain via Substrate.sol.

Originally targeted EpochAnchor.sol; updated in Phase 5.6a after the
pre-substrate contract nuke folded EpochAnchor's responsibilities
into the new substrate-native Substrate.sol.

Validates:
  1. Canonical encoding produces identical bytes across daemons given
     identical inputs.
  2. Anchor lands on chain; getAnchorByEpochId retrieves it correctly.
  3. Anchor chain integrity: prev_epoch_root and prev_anchor_hash
     mismatches revert.
  4. Empty epochs can be anchored (using EMPTY_EPOCH_ROOT).
  5. Three-daemon federation: same canonical close -> same encoded
     payload -> same payload_hash -> same agent_mint_cid; first
     submission wins, others get EpochAlreadyAnchored.
  6. Off-chain agent_mint blob is decodable via the CID and reproduces
     the original agent_mint dict.
"""

from __future__ import annotations

import hashlib
import json
import random
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
    encode_authoritative_payload,
    payload_hash,
)
from nodes.common.canonical_ordering import EMPTY_EPOCH_ROOT, canonical_order
from nodes.common.epoch_anchorer import (
    AnchorResult,
    EpochAnchorer,
    EpochAnchorerConfig,
)
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import federated_epoch_close


ARTIFACTS_DIR = Path("C:/code/autonet/artifacts/contracts")
EPOCH_ANCHOR_JSON = ARTIFACTS_DIR / "core/Substrate.sol/Substrate.json"


def _load_artifact(path: Path) -> Tuple[list, str]:
    if not path.exists():
        pytest.skip(f"missing artifact: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["abi"], data["bytecode"]


def _deploy(w3: Web3, deployer: str, abi: list, bytecode: str, *args) -> str:
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor(*args).transact({"from": deployer, "gas": 8_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1, f"deploy failed: {receipt}"
    return receipt.contractAddress


@pytest.fixture
def chain():
    """Fresh in-process EVM with EpochAnchor deployed."""
    abi, bytecode = _load_artifact(EPOCH_ANCHOR_JSON)
    w3 = Web3(EthereumTesterProvider())
    deployer = w3.eth.accounts[0]
    addr = _deploy(
        w3, deployer, abi, bytecode,
        deployer,  # treasury
        "0x0000000000000000000000000000000000000000",  # vaultMinter (off)
    )
    contract = w3.eth.contract(address=addr, abi=abi)
    return {
        "w3": w3,
        "abi": abi,
        "addr": addr,
        "contract": contract,
        "deployer": deployer,
    }


# ---------------------------------------------------------------------------
# Helpers: build a minimal federated close result
# ---------------------------------------------------------------------------


def _make_chain(rpb: str, kp: Keypair, agent_id: str, n_batches: int) -> List[EventBatch]:
    chain: List[EventBatch] = []
    prev = b""
    for i in range(1, n_batches + 1):
        ev = {
            "kind": "observation_added",
            "seq": 1,
            "author_agent": agent_id,
            "obs_id": f"obs_{agent_id}_{i}",
            "coords": [0.0, 0.0, 0.5, 0.0] + [0.0] * 1023 + [0.0],
            "label": f"{agent_id}_{i}",
        }
        # Keep coords length 1028 (4 charter + 1024 embedding).
        coords = [0.0] * (4 + 1024)
        coords[2] = 0.5  # intelligence axis
        coords[4 + i] = 0.5  # spread across embedding tail
        ev["coords"] = coords
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


def _make_close_result(
    rpb: str = "rpb_anchor_test",
    n_senders: int = 2,
    n_batches_each: int = 2,
    epoch_id: str = "e_alpha",
) -> Dict[str, Any]:
    senders = [Keypair.generate() for _ in range(n_senders)]
    batches: List[EventBatch] = []
    for i, kp in enumerate(senders):
        batches.extend(_make_chain(rpb, kp, f"agent_{i}", n_batches_each))
    canonical = canonical_order(batches)
    result = federated_epoch_close(canonical)
    result["epoch_id"] = epoch_id
    return result


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_encoding_is_byte_identical_across_calls():
    """Same payload + identifiers → same bytes, every call."""
    result = _make_close_result(epoch_id="e_enc")
    payload = result["authoritative_payload"]

    a = encode_authoritative_payload(
        payload,
        epoch_id="e_enc",
        epoch_root_hex=payload["epoch_root"],
        prev_epoch_root_hex="00" * 32,
    )
    b = encode_authoritative_payload(
        payload,
        epoch_id="e_enc",
        epoch_root_hex=payload["epoch_root"],
        prev_epoch_root_hex="00" * 32,
    )
    assert a == b
    # And the bytes parse back to the same dict (sanity).
    decoded = json.loads(a.decode("utf-8"))
    assert decoded["epoch_id"] == "e_enc"
    assert decoded["epoch_root"] == payload["epoch_root"]


def test_encoding_rejects_non_finite_floats():
    """NaN / inf in agent_mint must be rejected, not silently anchored."""
    payload = {
        "epoch_root": "00" * 32,
        "agent_mint": {"alice": float("inf")},
        "agent_novelty": {},
        "total_mint": 0.0,
        "total_novelty": 0.0,
        "n_batches": 0,
        "n_events": 0,
    }
    with pytest.raises(ValueError):
        encode_authoritative_payload(
            payload,
            epoch_id="e_bad",
            epoch_root_hex="00" * 32,
            prev_epoch_root_hex="00" * 32,
        )


def test_agent_mint_blob_round_trip():
    """The off-chain blob decodes back to the original mint dict."""
    mint = {"alice": 1.234567890123, "bob": 0.5, "carol_zzz": 7.7}
    blob = encode_agent_mint_blob(mint)
    decoded = decode_agent_mint_blob(blob)
    # Values are rounded to 10 decimals at encode time.
    assert decoded["alice"] == pytest.approx(round(1.234567890123, 10))
    assert decoded["bob"] == 0.5
    assert decoded["carol_zzz"] == 7.7
    # CID is sha256 hex.
    cid = cid_for_blob(blob)
    assert len(cid) == 64
    assert all(c in "0123456789abcdef" for c in cid)


# ---------------------------------------------------------------------------
# Anchor chain integrity
# ---------------------------------------------------------------------------


def test_first_anchor_lands_with_zero_prev_root_and_zero_prev_anchor(chain):
    """Initial anchor: latestEpochRoot and latestAnchorHash are zero
    when no anchors exist; the first anchor's prev_* must be zero."""
    w3 = chain["w3"]
    contract = chain["contract"]
    deployer = chain["deployer"]

    # Sanity preconditions.
    assert bytes(contract.functions.latestEpochRoot().call()) == b"\x00" * 32
    assert bytes(contract.functions.latestAnchorHash().call()) == b"\x00" * 32
    assert contract.functions.anchorCount().call() == 0

    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=w3,
        contract_abi=chain["abi"],
    )

    result = _make_close_result(epoch_id="e_first")
    submission = anchorer.anchor_close_result(result, from_address=deployer)
    assert submission.success, submission.error
    assert submission.epoch_id == "e_first"

    # On-chain: anchor count incremented, retrievable by epoch_id.
    assert contract.functions.anchorCount().call() == 1
    a = contract.functions.getAnchorByEpochId("e_first").call()
    assert a[0] == "e_first"          # epochId
    assert bytes(a[2]) == b"\x00" * 32  # prevEpochRoot
    assert bytes(a[3]) == b"\x00" * 32  # prevAnchorHash
    # CID is deterministic from the agent_mint blob.
    assert a[4] == submission.agent_mint_cid
    # payload_hash matches.
    assert bytes(a[5]) == bytes.fromhex(submission.payload_hash_hex)


def test_anchor_chain_rejects_wrong_prev_epoch_root(chain):
    """Forging a prev_epoch_root that doesn't match the on-chain
    latest reverts."""
    w3 = chain["w3"]
    contract = chain["contract"]
    deployer = chain["deployer"]
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=w3,
        contract_abi=chain["abi"],
    )

    # First anchor lands cleanly.
    res1 = _make_close_result(epoch_id="e_legit")
    s1 = anchorer.anchor_close_result(res1, from_address=deployer)
    assert s1.success

    # Now an attacker tries to submit a second anchor with
    # prev_epoch_root != latestEpochRoot. We bypass the anchorer's
    # automatic linkage and call submitAnchor directly with a
    # bogus prev_epoch_root.
    res2 = _make_close_result(epoch_id="e_attacker")
    payload = res2["authoritative_payload"]
    bogus_prev_root = b"\xaa" * 32
    bogus_prev_anchor = bytes(contract.functions.latestAnchorHash().call())
    encoded = encode_authoritative_payload(
        payload,
        epoch_id="e_attacker",
        epoch_root_hex=payload["epoch_root"],
        prev_epoch_root_hex=bogus_prev_root.hex(),
    )
    ph = payload_hash(encoded)

    # eth_tester doesn't raise on revert by default; we either get an
    # exception OR a status=0 receipt. Both count as "rejected".
    rejected = False
    try:
        tx = contract.functions.submitAnchor(
            "e_attacker",
            bytes.fromhex(payload["epoch_root"]),
            bogus_prev_root,
            bogus_prev_anchor,
            "0" * 64,
            ph,
            b"\x00" * 32,
        ).transact({"from": deployer, "gas": 600_000})
        receipt = w3.eth.wait_for_transaction_receipt(tx)
        if receipt.status != 1:
            rejected = True
    except Exception:
        rejected = True
    assert rejected, "PrevEpochRootMismatch was not enforced"
    # Anchor count unchanged.
    assert contract.functions.anchorCount().call() == 1


def test_anchor_chain_rejects_wrong_prev_anchor_hash(chain):
    """Same idea but tampering with the anchor-chain prev hash."""
    w3 = chain["w3"]
    contract = chain["contract"]
    deployer = chain["deployer"]
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=w3,
        contract_abi=chain["abi"],
    )
    res1 = _make_close_result(epoch_id="e_legit_2")
    s1 = anchorer.anchor_close_result(res1, from_address=deployer)
    assert s1.success

    res2 = _make_close_result(epoch_id="e_bad_anchor")
    payload = res2["authoritative_payload"]
    real_prev_root = bytes(contract.functions.latestEpochRoot().call())
    bogus_prev_anchor = b"\xff" * 32
    encoded = encode_authoritative_payload(
        payload,
        epoch_id="e_bad_anchor",
        epoch_root_hex=payload["epoch_root"],
        prev_epoch_root_hex=real_prev_root.hex(),
    )
    ph = payload_hash(encoded)

    rejected = False
    try:
        tx = contract.functions.submitAnchor(
            "e_bad_anchor",
            bytes.fromhex(payload["epoch_root"]),
            real_prev_root,
            bogus_prev_anchor,
            "0" * 64,
            ph,
            b"\x00" * 32,
        ).transact({"from": deployer, "gas": 600_000})
        receipt = w3.eth.wait_for_transaction_receipt(tx)
        if receipt.status != 1:
            rejected = True
    except Exception:
        rejected = True
    assert rejected, "PrevAnchorHashMismatch was not enforced"
    # Only the legit anchor remains.
    assert chain["contract"].functions.anchorCount().call() == 1


def test_duplicate_epoch_id_reverts(chain):
    """Two anchors for the same epoch_id: second reverts. This is
    Phase 5.5's collision behavior; fork resolution is later."""
    w3 = chain["w3"]
    deployer = chain["deployer"]
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=w3,
        contract_abi=chain["abi"],
    )

    res = _make_close_result(epoch_id="e_dup")
    s1 = anchorer.anchor_close_result(res, from_address=deployer)
    assert s1.success

    s2 = anchorer.anchor_close_result(res, from_address=deployer)
    assert not s2.success
    # Error string mentions the revert.
    assert "EpochAlreadyAnchored" in s2.error or "revert" in s2.error.lower()


# ---------------------------------------------------------------------------
# Empty epoch
# ---------------------------------------------------------------------------


def test_empty_epoch_can_be_anchored(chain):
    """An epoch with no senders accepted produces EMPTY_EPOCH_ROOT;
    it can be anchored cleanly so that the chain has a continuous
    record even of empty windows."""
    w3 = chain["w3"]
    deployer = chain["deployer"]
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=w3,
        contract_abi=chain["abi"],
    )

    canonical = canonical_order([])
    result = federated_epoch_close(canonical)
    result["epoch_id"] = "e_empty"
    assert result["epoch_root"] == EMPTY_EPOCH_ROOT.hex()
    assert result["agent_mint"] == {}

    submission = anchorer.anchor_close_result(result, from_address=deployer)
    assert submission.success, submission.error
    # The chain reader can distinguish "empty epoch" from
    # "uninitialized": EMPTY_EPOCH_ROOT != b"\x00" * 32.
    contract = chain["contract"]
    a = contract.functions.getAnchorByEpochId("e_empty").call()
    assert bytes(a[1]) == EMPTY_EPOCH_ROOT  # epochRoot field


# ---------------------------------------------------------------------------
# Federation: three daemons, same submission
# ---------------------------------------------------------------------------


def test_three_daemons_compute_same_payload_first_anchor_wins(chain):
    """Three daemons run the same federated close → same encoded
    payload bytes → same payload_hash → same CID. They each try to
    submit. The first lands; the other two get EpochAlreadyAnchored.

    This is the Phase 5.5 contract: anchoring is idempotent on the
    network — only one anchor per epoch_id, and it's content-
    determined."""
    w3 = chain["w3"]
    deployer = chain["deployer"]

    rpb = "rpb_3daemon"
    senders = [Keypair.generate() for _ in range(3)]
    chains = [
        _make_chain(rpb, kp, f"agent_{i}", 2)
        for i, kp in enumerate(senders)
    ]
    all_batches = [b for c in chains for b in c]

    rng = random.Random(2026)
    deliveries = [list(all_batches) for _ in range(3)]
    for d in deliveries:
        rng.shuffle(d)

    results = []
    for d in deliveries:
        canonical = canonical_order(d)
        r = federated_epoch_close(canonical)
        r["epoch_id"] = "e_federated"
        results.append(r)

    # Sanity: all three computed identical authoritative payloads.
    assert (
        results[0]["authoritative_payload"]
        == results[1]["authoritative_payload"]
        == results[2]["authoritative_payload"]
    )

    # Each daemon would encode the same payload identically.
    encoded = [
        encode_authoritative_payload(
            r["authoritative_payload"],
            epoch_id="e_federated",
            epoch_root_hex=r["authoritative_payload"]["epoch_root"],
            prev_epoch_root_hex="00" * 32,
        )
        for r in results
    ]
    assert encoded[0] == encoded[1] == encoded[2]
    # And the same CIDs.
    blobs = [encode_agent_mint_blob(r["agent_mint"]) for r in results]
    assert blobs[0] == blobs[1] == blobs[2]
    assert (
        cid_for_blob(blobs[0])
        == cid_for_blob(blobs[1])
        == cid_for_blob(blobs[2])
    )

    # Now actually submit. Three different submitter addresses.
    submitters = w3.eth.accounts[:3]
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=w3,
        contract_abi=chain["abi"],
    )
    submissions = []
    for sub_addr, r in zip(submitters, results):
        submissions.append(
            anchorer.anchor_close_result(r, from_address=sub_addr)
        )

    # First wins; rest get EpochAlreadyAnchored.
    assert submissions[0].success
    assert not submissions[1].success
    assert not submissions[2].success
    # The chain has exactly one anchor for this epoch.
    assert chain["contract"].functions.anchorCount().call() == 1


# ---------------------------------------------------------------------------
# Agent reads its mint via CID
# ---------------------------------------------------------------------------


def test_agent_reads_mint_from_cid_blob(chain):
    """An agent reads the on-chain anchor, fetches the off-chain
    blob (here passed in directly from the submission's
    ``agent_mint_blob``), looks up its mint amount."""
    w3 = chain["w3"]
    deployer = chain["deployer"]
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=w3,
        contract_abi=chain["abi"],
    )
    result = _make_close_result(epoch_id="e_lookup")
    submission = anchorer.anchor_close_result(result, from_address=deployer)
    assert submission.success

    # Agent reads the anchor for "e_lookup".
    contract = chain["contract"]
    a = contract.functions.getAnchorByEpochId("e_lookup").call()
    cid_on_chain = a[4]
    assert cid_on_chain == submission.agent_mint_cid

    # Agent fetches the blob (in production via P2P; here we have it
    # already in submission.agent_mint_blob). Verify the CID matches.
    blob = submission.agent_mint_blob
    assert cid_for_blob(blob) == cid_on_chain

    # Decode and look up the agent's mint.
    mint_map = decode_agent_mint_blob(blob)
    expected = result["agent_mint"]
    # Whatever agents are in the authoritative map decode back with
    # matching values. (Empty map is also valid — the federation may
    # cluster all events into one agent's contributions, or none if
    # all canonical batches dropped.)
    assert set(mint_map.keys()) == set(expected.keys())
    for agent_id in expected:
        assert mint_map[agent_id] == pytest.approx(expected[agent_id])
