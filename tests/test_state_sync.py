"""Network state sync: canonical world checkpoints + anchored catch-up.

Validates the rejoining-daemon story end to end:

  - every daemon's CanonicalWorldTracker computes the same checkpoint
    blob (cid consensus),
  - the cid rides the authoritative payload (schema 2) onto the chain
    anchor,
  - a fresh daemon catches up via chain reads + blob fetches alone and
    lands on EXACTLY the canonical world,
  - integrity violations (wrong blobs, contradicting epoch_root) are
    rejected, missing data degrades to None (not a crash),
  - the caught-up world installs as a local checkpoint so the normal
    boot path picks it up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

eth_tester = pytest.importorskip("eth_tester")

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

from nodes.common.blob_resolver import InMemoryBlobResolver
from nodes.common.canonical_ordering import canonical_order
from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import federated_epoch_close
from nodes.common.state_sync import (
    CanonicalWorldTracker,
    catch_up_from_chain,
    decode_world_checkpoint,
    encode_world_checkpoint,
    install_as_local_checkpoint,
)
from nodes.common.world_persistence import PersistenceConfig, WorldPersistence
from world_model.generalized import worlds_equal

ARTIFACT = Path("C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json")
EMBED_DIM = 8
RPB = "rpb_state_sync"


@pytest.fixture
def chain():
    if not ARTIFACT.exists():
        pytest.skip(f"missing artifact: {ARTIFACT}")
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    w3 = Web3(EthereumTesterProvider())
    deployer = w3.eth.accounts[0]
    contract = w3.eth.contract(abi=data["abi"], bytecode=data["bytecode"])
    tx = contract.constructor(deployer).transact({"from": deployer, "gas": 8_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    return {
        "w3": w3,
        "abi": data["abi"],
        "addr": receipt.contractAddress,
        "contract": w3.eth.contract(address=receipt.contractAddress, abi=data["abi"]),
        "deployer": deployer,
    }


def _batches(kp: Keypair, agent_id: str, n: int, start_seq: int = 1) -> List[EventBatch]:
    out: List[EventBatch] = []
    prev = b""
    for i in range(n):
        coords = [0.0] * (6 + EMBED_DIM)
        coords[i % 6] = 0.6 if i % 2 == 0 else -0.5
        coords[6 + (i % EMBED_DIM)] = 0.4
        ev = {
            "kind": "observation_added",
            "seq": 1,
            "author_agent": agent_id,
            "obs_id": f"obs_{agent_id}_{start_seq + i}",
            "coords": coords,
            "label": f"{agent_id}_{start_seq + i}",
        }
        b = EventBatch(
            rpb_address=RPB,
            sender_pubkey=kp.public_key,
            batch_seq=start_seq + i,
            events=[ev],
            prev_batch_hash=prev,
            timestamp=1_700_000_000.0 + start_seq + i,
        )
        out.append(b)
        prev = b.content_hash()
    return out


def _close_one_epoch(tracker, anchorer, resolver, kp, epoch_id, start_seq=1):
    """Replicate the FederatedCloseDriver + ChainSubmissionDriver flow
    for one epoch: canonical order -> federated close -> tracker ->
    world_cid into payload -> anchor -> publish blobs.
    """
    canonical = canonical_order(_batches(kp, "agent_a", 2, start_seq=start_seq))
    close_result = federated_epoch_close(canonical, embedding_dim=EMBED_DIM)
    close_result["epoch_id"] = epoch_id

    ckpt = tracker.on_close(
        epoch_id,
        str(close_result.get("epoch_root", "")),
        [list(b.events or []) for b in canonical.ordered_batches],
    )
    close_result["authoritative_payload"]["world_cid"] = ckpt.cid

    anchor = anchorer.anchor_close_result(close_result)
    assert anchor.success, anchor.error
    resolver.put(anchor.payload_bytes)   # cid == on-chain payloadHash
    resolver.put(ckpt.blob)
    return ckpt, anchor


@pytest.fixture
def synced_network(chain):
    """A network that has closed two epochs with state sync active."""
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=chain["w3"], contract_abi=chain["abi"],
    )
    anchorer._set_submitter_address(chain["deployer"])
    resolver = InMemoryBlobResolver()
    tracker = CanonicalWorldTracker(embedding_dim=EMBED_DIM)
    kp = Keypair.generate()

    ckpt1, _ = _close_one_epoch(tracker, anchorer, resolver, kp, "e_sync_1", start_seq=1)
    ckpt2, _ = _close_one_epoch(tracker, anchorer, resolver, kp, "e_sync_2", start_seq=10)
    return {
        "chain": chain,
        "resolver": resolver,
        "tracker": tracker,
        "ckpts": [ckpt1, ckpt2],
    }


# ---------------------------------------------------------------------------
# Tracker determinism
# ---------------------------------------------------------------------------


def test_trackers_agree_on_checkpoint_cid():
    kp = Keypair.generate()
    canonical = canonical_order(_batches(kp, "agent_a", 3))
    batch_events = [list(b.events or []) for b in canonical.ordered_batches]

    t1 = CanonicalWorldTracker(embedding_dim=EMBED_DIM)
    t2 = CanonicalWorldTracker(embedding_dim=EMBED_DIM)
    c1 = t1.on_close("e_x", "00" * 32, batch_events)
    c2 = t2.on_close("e_x", "00" * 32, batch_events)
    assert c1.cid == c2.cid
    assert c1.blob == c2.blob
    assert worlds_equal(t1.world, t2.world)


def test_tracker_resumed_from_checkpoint_continues_identically():
    kp = Keypair.generate()
    first = [list(b.events or []) for b in
             canonical_order(_batches(kp, "agent_a", 2)).ordered_batches]
    second = [list(b.events or []) for b in
              canonical_order(_batches(kp, "agent_a", 2, start_seq=10)).ordered_batches]

    full = CanonicalWorldTracker(embedding_dim=EMBED_DIM)
    c1 = full.on_close("e_1", "00" * 32, first)
    c2_full = full.on_close("e_2", "11" * 32, second)

    resumed = CanonicalWorldTracker.from_checkpoint(
        decode_world_checkpoint(c1.blob), c1.cid, embedding_dim=EMBED_DIM,
    )
    c2_resumed = resumed.on_close("e_2", "11" * 32, second)

    assert c2_resumed.cid == c2_full.cid
    assert worlds_equal(resumed.world, full.world)


def test_checkpoint_chains_via_prev_world_cid():
    kp = Keypair.generate()
    t = CanonicalWorldTracker(embedding_dim=EMBED_DIM)
    evs = [list(b.events or []) for b in
           canonical_order(_batches(kp, "a", 1)).ordered_batches]
    c1 = t.on_close("e_1", "00" * 32, evs)
    c2 = t.on_close("e_2", "11" * 32, evs)
    assert decode_world_checkpoint(c1.blob)["prev_world_cid"] == ""
    assert decode_world_checkpoint(c2.blob)["prev_world_cid"] == c1.cid


# ---------------------------------------------------------------------------
# Catch-up
# ---------------------------------------------------------------------------


def test_catch_up_restores_latest_canonical_world(synced_network):
    result = catch_up_from_chain(
        synced_network["chain"]["contract"], synced_network["resolver"],
    )
    assert result is not None
    assert result.epoch_id == "e_sync_2"
    assert result.world_cid == synced_network["ckpts"][1].cid
    assert result.prev_world_cid == synced_network["ckpts"][0].cid
    assert worlds_equal(result.world, synced_network["tracker"].world)


def test_catch_up_none_when_no_anchors(chain):
    assert catch_up_from_chain(chain["contract"], InMemoryBlobResolver()) is None


def test_catch_up_none_when_blobs_missing(synced_network):
    # Chain has anchors but this "daemon" can't reach any blob.
    assert catch_up_from_chain(
        synced_network["chain"]["contract"], InMemoryBlobResolver(),
    ) is None


def test_catch_up_rejects_contradicting_epoch_root(synced_network):
    # Forge a world blob whose declared epoch_root doesn't match the
    # anchor, re-point the payload at it, and serve both. The payload
    # tampering breaks the payloadHash check first — so instead tamper
    # the world blob reference target only: serve a blob at the right
    # cid is impossible (cid = hash), so simulate a confused resolver
    # returning wrong bytes for the world cid.
    resolver = synced_network["resolver"]
    real = synced_network["ckpts"][1]
    forged = encode_world_checkpoint(
        synced_network["tracker"].world,
        epoch_id="e_sync_2",
        epoch_root_hex="ff" * 32,           # contradicts the anchor
        prev_world_cid=real.prev_world_cid,
    )

    class LyingResolver:
        def get(self, cid):
            if cid == real.cid:
                return forged                # wrong bytes for this cid
            return resolver.get(cid)

    with pytest.raises(ValueError):
        catch_up_from_chain(synced_network["chain"]["contract"], LyingResolver())


def test_catch_up_none_for_pre_state_sync_anchor(chain):
    """Anchors whose payload has no world_cid (schema 1 era) degrade
    to None instead of crashing."""
    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=chain["w3"], contract_abi=chain["abi"],
    )
    anchorer._set_submitter_address(chain["deployer"])
    kp = Keypair.generate()
    canonical = canonical_order(_batches(kp, "agent_a", 1))
    close_result = federated_epoch_close(canonical, embedding_dim=EMBED_DIM)
    close_result["epoch_id"] = "e_old"
    # No world_cid injected -> encodes as "".
    anchor = anchorer.anchor_close_result(close_result)
    assert anchor.success, anchor.error
    resolver = InMemoryBlobResolver()
    resolver.put(anchor.payload_bytes)

    assert catch_up_from_chain(chain["contract"], resolver) is None


# ---------------------------------------------------------------------------
# Local install (boot integration)
# ---------------------------------------------------------------------------


def test_install_as_local_checkpoint_boots_canonical_world(synced_network, tmp_path):
    result = catch_up_from_chain(
        synced_network["chain"]["contract"], synced_network["resolver"],
    )
    persistence = WorldPersistence(
        PersistenceConfig(rpb_address="ckpt-sync", data_root=tmp_path),
    )
    install_as_local_checkpoint(persistence, result)

    restored = persistence.try_restore()
    assert restored is not None
    assert restored.from_checkpoint
    assert worlds_equal(restored.world, synced_network["tracker"].world)
    persistence.close()


def test_install_refuses_nonempty_local_log(synced_network, tmp_path):
    result = catch_up_from_chain(
        synced_network["chain"]["contract"], synced_network["resolver"],
    )
    persistence = WorldPersistence(
        PersistenceConfig(rpb_address="ckpt-busy", data_root=tmp_path),
    )
    persistence.append_events([{
        "kind": "observation_added", "seq": 1, "author_agent": "x",
        "obs_id": "obs_local", "coords": [0.1] * 6, "label": "local",
    }])
    with pytest.raises(RuntimeError):
        install_as_local_checkpoint(persistence, result)
    persistence.close()
