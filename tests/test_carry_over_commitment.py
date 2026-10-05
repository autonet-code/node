"""Verifiable carry-over commitment (authoritative payload schema 4).

Each federated close serializes the five tool maps it produced (the next
close's inputs) into one canonical bundle blob; its cid rides the payload
as ``carry_cid``. A daemon that first boots after anchors exist fetches
the bundle from peers, verifies it against the on-chain payloadHash ->
carry_cid chain, installs it, and closes bit-identically to an always-on
daemon, without trusting the peer that served it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from nodes.common.authoritative_encoding import (
    CARRY_KEYS,
    SCHEMA_VERSION,
    cid_for_blob,
    decode_carry_bundle,
    encode_authoritative_payload,
    encode_carry_bundle,
)
from nodes.common.event_gossip import EventBatch, EventGossip, Keypair
from nodes.common.federated_close_driver import FederatedCloseDriver

EMBED_DIM = 8
DIGEST = "ab" * 32
AUTHOR = "toolsmith"
ROOT_A = "11" * 32
ROOT_B = "22" * 32


# ---------------------------------------------------------------------------
# Event fixtures: one tool registered in epoch 1, reviewed every epoch
# ---------------------------------------------------------------------------


def _coords(axis: int = 4) -> List[float]:
    out = [0.0] * (6 + EMBED_DIM)
    out[axis] = 0.8
    out[6 + 3] = 0.5
    return out


def _registration(seq: int) -> Dict[str, Any]:
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": AUTHOR,
        "tendency_id": "correctness",
        "parent_id": "solver_root",
        "node_id": f"tm_{DIGEST[:12]}",
        "position": "pro",
        "coords": _coords(),
        "polarity_axis": _coords(),
        "content": "tool: does a thing",
        "author_post": True,
        "artifact_digest": DIGEST,
        "manifest_meta": {"trust_class": "pinned", "author": AUTHOR},
    }


def _review(seq: int, caller: str, epoch: int, axes: Dict[str, float]) -> Dict[str, Any]:
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": caller,
        "manifest_digest": DIGEST,
        "tool_author": AUTHOR,
        "receipt_digest": f"{epoch:02d}{seq:02d}" * 16,
        "ok": True,
        "fee_atn": 0.0,
        "attested": True,
        "score": 0.8,
        "axes": dict(axes),
    }


def _chain(events: List[Dict[str, Any]], kp: Keypair, t0: float) -> List[EventBatch]:
    out: List[EventBatch] = []
    prev = b""
    for i, ev in enumerate(events, start=1):
        b = EventBatch(
            rpb_address="rpb_carry",
            sender_pubkey=kp.public_key,
            batch_seq=i,
            events=[ev],
            prev_batch_hash=prev,
            timestamp=t0 + i,
        )
        out.append(b)
        prev = b.content_hash()
    return out


def _epoch_batches(epoch: int) -> List[EventBatch]:
    t0 = 1_700_000_000.0 + epoch * 1000
    batches: List[EventBatch] = []
    if epoch == 1:
        batches += _chain([_registration(1)], Keypair.generate(), t0)
    # Reviewers disagree a bit so positions, the review book and (with
    # enough mass) credibility all move.
    reviews = {
        "rev_a": {"correctness": 0.9, "simplicity": 0.4},
        "rev_b": {"correctness": 0.7 - 0.1 * epoch, "simplicity": -0.2},
        "rev_c": {"correctness": -0.8, "simplicity": 0.1 * epoch},
    }
    for name, axes in sorted(reviews.items()):
        batches += _chain(
            [_review(s, name, epoch, axes) for s in (1, 2)],
            Keypair.generate(), t0 + 100,
        )
    return batches


EPOCHS = {n: _epoch_batches(n) for n in range(1, 8)}


def _gossip(batches: List[EventBatch]) -> MagicMock:
    g = MagicMock(spec=EventGossip)
    g.drain_epoch_batches.return_value = list(batches)
    senders = sorted({b.sender_pubkey for b in batches})
    g.known_senders.return_value = senders
    g.sender_pubkey = senders[0] if senders else b""
    return g


def _driver(state_dir: Path, voice_source=None, enforce=False) -> FederatedCloseDriver:
    state_dir.mkdir(parents=True, exist_ok=True)
    d = FederatedCloseDriver(
        gossip=_gossip([]),
        embedding_dim=EMBED_DIM,
        **{f"{k}_path": state_dir / f"{k}.json" for k in CARRY_KEYS},
    )
    d.enforce_preconditions = enforce
    d.voice_source = voice_source
    return d


def _run(d: FederatedCloseDriver, epoch: int):
    d.gossip = _gossip(EPOCHS[epoch])
    return d.run({"epoch_id": f"e{epoch}"})


def _payload_bytes(fed) -> bytes:
    payload = fed.close_result["authoritative_payload"]
    return encode_authoritative_payload(
        payload, epoch_id=fed.epoch_id,
        epoch_root_hex=payload["epoch_root"], prev_epoch_root_hex=ROOT_A,
    )


# ---------------------------------------------------------------------------
# Bundle determinism + encoding
# ---------------------------------------------------------------------------


def test_two_daemons_produce_identical_bundle(tmp_path):
    a = _driver(tmp_path / "a")
    b = _driver(tmp_path / "b")
    for n in (1, 2, 3):
        fa, fb = _run(a, n), _run(b, n)
        assert fa.carry_bundle_blob == fb.carry_bundle_blob
        assert fa.carry_cid == fb.carry_cid == cid_for_blob(fa.carry_bundle_blob)
        assert fa.close_result["authoritative_payload"]["carry_cid"] == fa.carry_cid
        assert _payload_bytes(fa) == _payload_bytes(fb)
    # The fixture actually exercises the carried maps.
    carry = decode_carry_bundle(fa.carry_bundle_blob)["carry"]
    assert carry["tool_registrations"] and carry["tool_positions"]
    assert carry["tool_review_book"]


def test_bundle_bytes_ignore_key_insertion_order(tmp_path):
    d = _driver(tmp_path)
    fed = _run(d, 1)
    carry = decode_carry_bundle(fed.carry_bundle_blob)["carry"]

    def _reverse(x):
        if isinstance(x, dict):
            return {k: _reverse(x[k]) for k in reversed(list(x))}
        return x

    scrambled = {k: _reverse(carry[k]) for k in reversed(CARRY_KEYS)}
    assert encode_carry_bundle(scrambled, epoch_id="e1") == fed.carry_bundle_blob


def test_bundle_round_trips_exact_floats():
    carry = {k: {} for k in CARRY_KEYS}
    carry["tool_positions"] = {DIGEST: {"head": [0.1 + 0.2, -0.0, 1e-9],
                                        "mass": [1.0, 2.5, 3.0]}}
    carry["tool_credibility"] = {"h": 0.123456789}
    blob = encode_carry_bundle(carry, epoch_id="e9")
    out = decode_carry_bundle(blob)
    assert out["epoch_id"] == "e9"
    assert out["carry"]["tool_positions"][DIGEST]["head"][0] == 0.1 + 0.2
    assert out["carry"] == carry


def test_bundle_rejects_nan_and_missing_maps():
    carry = {k: {} for k in CARRY_KEYS}
    carry["tool_credibility"] = {"h": float("nan")}
    with pytest.raises(ValueError):
        encode_carry_bundle(carry, epoch_id="e1")
    with pytest.raises(ValueError):
        encode_carry_bundle({"tool_registrations": {}}, epoch_id="e1")


def test_decode_rejects_non_canonical_or_malformed():
    carry = {k: {} for k in CARRY_KEYS}
    blob = encode_carry_bundle(carry, epoch_id="e1")
    pretty = json.dumps(json.loads(blob), indent=1).encode()
    with pytest.raises(ValueError):
        decode_carry_bundle(pretty)                      # same value, other bytes
    wrong = json.loads(blob)
    wrong["schema"] = 99
    with pytest.raises(ValueError):
        decode_carry_bundle(json.dumps(wrong, sort_keys=True,
                                       separators=(",", ":")).encode())
    missing = json.loads(blob)
    del missing["carry"]["tool_vetting"]
    with pytest.raises(ValueError):
        decode_carry_bundle(json.dumps(missing, sort_keys=True,
                                       separators=(",", ":")).encode())
    with pytest.raises(ValueError):
        decode_carry_bundle(b"not json")


def test_payload_encodes_carry_cid_last_and_schema_4():
    payload = {
        "epoch_root": ROOT_B,
        "agent_mint": {"bob": 1.0, "alice": 2.0},
        "agent_novelty": {},
        "total_mint": 3.0,
        "total_novelty": 0.0,
        "world_cid": "aa" * 32,
        "carry_cid": "cc" * 32,
    }
    raw = encode_authoritative_payload(
        payload, epoch_id="e1", epoch_root_hex=ROOT_B, prev_epoch_root_hex=ROOT_A)
    decoded = json.loads(raw)
    assert SCHEMA_VERSION == 4
    assert decoded["schema"] == 4
    assert decoded["carry_cid"] == "cc" * 32
    assert list(decoded)[-2:] == ["world_cid", "carry_cid"]
    # Absent carry_cid encodes as "" (no tracker / library use).
    del payload["carry_cid"]
    raw2 = encode_authoritative_payload(
        payload, epoch_id="e1", epoch_root_hex=ROOT_B, prev_epoch_root_hex=ROOT_A)
    assert json.loads(raw2)["carry_cid"] == ""
    assert raw2 != raw


# ---------------------------------------------------------------------------
# Joiner path against a real (eth_tester) Substrate
# ---------------------------------------------------------------------------

ARTIFACT = Path("C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json")


@pytest.fixture
def chain():
    pytest.importorskip("eth_tester")
    from web3 import Web3
    from web3.providers.eth_tester import EthereumTesterProvider

    if not ARTIFACT.exists():
        pytest.skip(f"missing artifact: {ARTIFACT}")
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    w3 = Web3(EthereumTesterProvider())
    deployer = w3.eth.accounts[0]
    contract = w3.eth.contract(abi=data["abi"], bytecode=data["bytecode"])
    tx = contract.constructor(
        deployer, "0x0000000000000000000000000000000000000000",
        "0x0000000000000000000000000000000000000000",
    ).transact({"from": deployer, "gas": 8_000_000})
    receipt = w3.eth.wait_for_transaction_receipt(tx)
    assert receipt.status == 1
    return {
        "w3": w3,
        "abi": data["abi"],
        "addr": receipt.contractAddress,
        "contract": w3.eth.contract(address=receipt.contractAddress, abi=data["abi"]),
        "deployer": deployer,
    }


@pytest.fixture
def network(chain, tmp_path):
    """An always-on daemon that closed epochs 1..3, anchoring each close
    through ChainSubmissionDriver (which publishes payload + carry blobs)."""
    from nodes.common.blob_resolver import InMemoryBlobResolver
    from nodes.common.chain_submission_driver import (
        ChainSubmissionConfig,
        ChainSubmissionDriver,
    )
    from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig

    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=chain["w3"], contract_abi=chain["abi"],
    )
    anchorer._set_submitter_address(chain["deployer"])
    resolver = InMemoryBlobResolver()
    csd = ChainSubmissionDriver(
        config=ChainSubmissionConfig(substrate_address=chain["addr"]),
        agent_chain_resolver=lambda: [],
        blob_resolver=resolver,
        anchorer=anchorer,
    )

    def voice():
        # Mirrors read_voice_state: snapshot_block None <=> no anchors.
        n = int(chain["contract"].functions.anchorCount().call())
        return {"owner_map": {}, "voice_weights": {},
                "snapshot_block": None if n == 0 else n}

    always_on = _driver(tmp_path / "always_on", voice_source=voice, enforce=True)
    for n in (1, 2, 3):
        fed = _run(always_on, n)
        assert fed is not None
        fed.is_winner = True     # single always-on daemon anchors
        out = csd.handle_federated_close(fed)
        assert out["anchored"], out
        assert out["carry_cid"] == fed.carry_cid
        assert resolver.has(fed.carry_cid)
    return {"chain": chain, "resolver": resolver, "csd": csd,
            "voice": voice, "always_on": always_on}


def test_joiner_fetches_verifies_installs_and_closes_identically(network, tmp_path):
    from nodes.common.state_sync import fetch_carry_over_from_chain

    contract = network["chain"]["contract"]
    resolver = network["resolver"]
    joiner_dir = tmp_path / "joiner"
    joiner = _driver(joiner_dir, voice_source=network["voice"], enforce=True)
    joiner.carry_source = lambda: fetch_carry_over_from_chain(contract, resolver)
    # Booting mid-network: observe one epoch (3) before joining.
    assert _run(joiner, 3) is None
    assert not (joiner_dir / "tool_positions.json").exists()

    fed_on = _run(network["always_on"], 4)
    fed_join = _run(joiner, 4)
    assert fed_on is not None and fed_join is not None

    # Installed carry == what the always-on daemon carried into epoch 4.
    for k in CARRY_KEYS:
        assert (joiner_dir / f"{k}.json").exists()
    # Bit-identical close: payload bytes, mint, and the next bundle.
    assert _payload_bytes(fed_join) == _payload_bytes(fed_on)
    assert fed_join.close_result["agent_mint"] == fed_on.close_result["agent_mint"]
    assert fed_join.carry_bundle_blob == fed_on.carry_bundle_blob
    assert fed_join.carry_cid == fed_on.carry_cid
    # Control: without the carry-over, the same close forks.
    naive = _run(_driver(tmp_path / "naive"), 4)
    assert naive.carry_cid != fed_on.carry_cid
    # And it stays in lockstep afterwards.
    assert _payload_bytes(_run(joiner, 5)) == _payload_bytes(_run(network["always_on"], 5))


def test_joiner_refuses_bundle_when_anchor_lags(network, tmp_path, caplog):
    """The always-on daemon closes epoch 4 but the winner never anchors it:
    the latest anchor (3) is stale relative to the joiner's next close (5),
    so the joiner must not install it."""
    from nodes.common.state_sync import fetch_carry_over_from_chain

    contract = network["chain"]["contract"]
    resolver = network["resolver"]
    assert _run(network["always_on"], 4) is not None        # NOT anchored
    joiner_dir = tmp_path / "joiner"
    joiner = _driver(joiner_dir, voice_source=network["voice"], enforce=True)
    joiner.carry_source = lambda: fetch_carry_over_from_chain(contract, resolver)
    assert _run(joiner, 4) is None                          # observe 4
    with caplog.at_level(logging.WARNING):
        assert _run(joiner, 5) is None
    assert "is not the epoch observed immediately before this close" in caplog.text
    assert not any((joiner_dir / f"{k}.json").exists() for k in CARRY_KEYS)
    # Once epoch 5 is anchored, the joiner (having observed 5) joins at 6
    # in lockstep with the always-on daemon.
    fed5 = _run(network["always_on"], 5)
    fed5.is_winner = True
    assert network["csd"].handle_federated_close(fed5)["anchored"]
    assert _payload_bytes(_run(joiner, 6)) == _payload_bytes(_run(network["always_on"], 6))


def test_joiner_skips_epoch_peers_already_anchored(network, tmp_path):
    """Peers anchor epoch 4 before the joiner runs its own close of 4: the
    latest bundle is epoch 4's OUTPUT, so the joiner must not close 4 on
    it; it joins at 5 and matches the always-on daemon."""
    from nodes.common.state_sync import fetch_carry_over_from_chain

    contract = network["chain"]["contract"]
    resolver = network["resolver"]
    joiner = _driver(tmp_path / "joiner", voice_source=network["voice"], enforce=True)
    joiner.carry_source = lambda: fetch_carry_over_from_chain(contract, resolver)
    assert _run(joiner, 3) is None                          # observe 3
    fed4 = _run(network["always_on"], 4)
    fed4.is_winner = True
    assert network["csd"].handle_federated_close(fed4)["anchored"]
    assert _run(joiner, 4) is None                          # anchor is 4, not 3
    assert _payload_bytes(_run(joiner, 5)) == _payload_bytes(_run(network["always_on"], 5))


def test_fetch_reports_anchor_and_carry(network):
    from nodes.common.state_sync import fetch_carry_over_from_chain

    fetch = fetch_carry_over_from_chain(
        network["chain"]["contract"], network["resolver"])
    assert fetch.status == "ok"
    assert fetch.epoch_id == "e3"
    assert fetch.payload_schema == 4
    assert fetch.carry == {
        k: json.loads(json.dumps(v))
        for k, v in network["always_on"].carry_over_maps().items()
    }


def test_tampered_bundle_is_rejected(network, tmp_path, caplog):
    from nodes.common.state_sync import fetch_carry_over_from_chain

    contract = network["chain"]["contract"]
    real = network["resolver"]
    carry_cid = fetch_carry_over_from_chain(contract, real).carry_cid
    good = real.get(carry_cid)
    bundle = decode_carry_bundle(good)
    bundle["carry"]["tool_credibility"] = {"attacker": 1.0}
    forged = encode_carry_bundle(bundle["carry"], epoch_id=bundle["epoch_id"])

    class LyingPeer:
        def get(self, cid):
            return forged if cid == carry_cid else real.get(cid)

    with pytest.raises(ValueError, match="hash mismatch"):
        fetch_carry_over_from_chain(contract, LyingPeer())

    joiner_dir = tmp_path / "joiner"
    joiner = _driver(joiner_dir, voice_source=network["voice"], enforce=True)
    joiner.carry_source = lambda: fetch_carry_over_from_chain(contract, LyingPeer())
    assert _run(joiner, 3) is None
    with caplog.at_level(logging.WARNING):
        assert _run(joiner, 4) is None
    assert "REJECTED anchored carry-over bundle" in caplog.text
    assert not any((joiner_dir / f"{k}.json").exists() for k in CARRY_KEYS)


def test_joiner_observes_until_bundle_retrievable(network, tmp_path):
    from nodes.common.blob_resolver import InMemoryBlobResolver
    from nodes.common.state_sync import fetch_carry_over_from_chain

    contract = network["chain"]["contract"]
    real = network["resolver"]
    carry_cid = fetch_carry_over_from_chain(contract, real).carry_cid
    partial = InMemoryBlobResolver()
    for cid in list(real._store):
        if cid != carry_cid:
            partial.put(real.get(cid))

    joiner = _driver(tmp_path / "joiner", voice_source=network["voice"], enforce=True)
    joiner.carry_source = lambda: fetch_carry_over_from_chain(contract, partial)
    assert _run(joiner, 3) is None
    assert _run(joiner, 4) is None
    assert fetch_carry_over_from_chain(contract, partial).status == "unavailable"


def test_pre_schema_anchor_keeps_refusing(chain, tmp_path, caplog):
    """An anchor whose payload has no carry_cid (schema <= 3 era, or a
    close anchored without the driver) cannot be joined trustlessly."""
    from nodes.common.blob_resolver import InMemoryBlobResolver
    from nodes.common.canonical_ordering import canonical_order
    from nodes.common.epoch_anchorer import EpochAnchorer, EpochAnchorerConfig
    from nodes.common.federated_reconcile import federated_epoch_close
    from nodes.common.state_sync import fetch_carry_over_from_chain

    anchorer = EpochAnchorer(
        config=EpochAnchorerConfig(epoch_anchor_address=chain["addr"]),
        web3=chain["w3"], contract_abi=chain["abi"],
    )
    anchorer._set_submitter_address(chain["deployer"])
    close = federated_epoch_close(canonical_order(EPOCHS[1]), embedding_dim=EMBED_DIM)
    close["epoch_id"] = "e_old"
    anchor = anchorer.anchor_close_result(close)
    assert anchor.success, anchor.error
    resolver = InMemoryBlobResolver()
    resolver.put(anchor.payload_bytes)

    fetch = fetch_carry_over_from_chain(chain["contract"], resolver)
    assert fetch.status == "pre_schema" and fetch.epoch_id == "e_old"

    joiner = _driver(
        tmp_path / "joiner", enforce=True,
        voice_source=lambda: {"owner_map": {}, "voice_weights": {},
                              "snapshot_block": 1},
    )
    joiner.carry_source = lambda: fetch_carry_over_from_chain(chain["contract"], resolver)
    assert _run(joiner, 1) is None
    with caplog.at_level(logging.WARNING):
        assert _run(joiner, 2) is None
    assert "predates carry-over commitments" in caplog.text


def test_fetch_genesis_status(chain):
    from nodes.common.blob_resolver import InMemoryBlobResolver
    from nodes.common.state_sync import fetch_carry_over_from_chain

    assert fetch_carry_over_from_chain(
        chain["contract"], InMemoryBlobResolver()).status == "genesis"
