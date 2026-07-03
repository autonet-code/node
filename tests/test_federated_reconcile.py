"""Phase 5.4 tests: bit-identical agent_mint maps across daemons.

Determinism contract: given the same canonical batch sequence, every
honest daemon's ``federated_epoch_close`` produces the SAME
``agent_mint`` dict at the byte level — same keys in the same order,
same values to the last bit.

That's the property that lets agents read a single authoritative
attribution from chain (Phase 5.5/5.6) without disputes: every
daemon agrees, no reconciliation needed.

Tested:

  1. Three daemons receive the same batches in different arrival
     orders, all canonicalize, all run federated_epoch_close,
     all produce dict-equal agent_mint maps and identical epoch_roots.
  2. The serialized JSON of the agent_mint map is byte-identical
     across daemons (key order included).
  3. Three daemons receive a mix of honest + malformed (broken-chain)
     senders. Canonical ordering drops the malformed sender; all three
     daemons still produce the same map, with the malformed sender
     contributing zero.
  4. agent_mint with no contributors is empty (not {"_unattributed":
     <something>}).
  5. Float determinism: a value computed from many small adds matches
     bit-for-bit between two reconciliation runs (within the
     output_decimals rounding).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pytest

from nodes.common.canonical_ordering import (
    EMPTY_EPOCH_ROOT,
    canonical_order,
)
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import (
    OUTPUT_DECIMALS,
    federated_epoch_close,
    federated_reconcile_epoch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coord(charter, embed_idx, mag):
    out = list(charter) + [0.0] * 1024
    out[4 + embed_idx] = mag
    return tuple(out)


def _obs_event(seq: int, agent_id: str, charter, embed_idx: int, label: str) -> Dict[str, Any]:
    return {
        "kind": "observation_added",
        "seq": seq,
        "author_agent": agent_id,
        "obs_id": f"obs_{label}",
        "coords": list(_coord(charter, embed_idx, 0.6)),
        "label": label,
    }


def _make_chain(
    rpb: str,
    keypair: Keypair,
    spec: List[Dict[str, Any]],
    agent_id: str,
) -> List[EventBatch]:
    """Build a signed-shaped batch chain for one sender from a list of
    per-batch specs ``[{charter: tuple, embed_idx: int, label: str}, ...]``.
    Each spec becomes one batch with one observation event."""
    chain: List[EventBatch] = []
    prev_hash = b""
    for i, s in enumerate(spec, start=1):
        ev = _obs_event(seq=1, agent_id=agent_id,
                        charter=s["charter"],
                        embed_idx=s["embed_idx"],
                        label=s["label"])
        b = EventBatch(
            rpb_address=rpb,
            sender_pubkey=keypair.public_key,
            batch_seq=i,
            events=[ev],
            prev_batch_hash=prev_hash,
            timestamp=1_700_000_000.0 + i,
        )
        chain.append(b)
        prev_hash = b.content_hash()
    return chain


# ---------------------------------------------------------------------------
# Bit-identical determinism
# ---------------------------------------------------------------------------


def test_three_daemons_produce_byte_identical_maps():
    """Same set of batches, three different arrival orders, three
    daemons. All three federated_epoch_close calls produce the SAME
    agent_mint dict at the byte level (same keys, same values, same
    insertion order)."""
    rpb = "rpb_fed_a"
    kp_alice = Keypair.generate()
    kp_bob = Keypair.generate()
    chain_a = _make_chain(rpb, kp_alice, [
        {"charter": (0.6, 0.0, 0.0, 0.0), "embed_idx": 1, "label": "a1"},
        {"charter": (0.0, 0.0, 0.6, 0.0), "embed_idx": 2, "label": "a2"},
    ], agent_id="alice")
    chain_b = _make_chain(rpb, kp_bob, [
        {"charter": (0.0, 0.6, 0.0, 0.0), "embed_idx": 3, "label": "b1"},
    ], agent_id="bob")

    rng = random.Random(7)
    arrivals = [list(chain_a + chain_b) for _ in range(3)]
    for a in arrivals:
        rng.shuffle(a)

    canonicals = [canonical_order(a) for a in arrivals]
    # Sanity: all three produced the same canonical sequence.
    seq_hashes = [
        [b.content_hash() for b in c.ordered_batches] for c in canonicals
    ]
    assert seq_hashes[0] == seq_hashes[1] == seq_hashes[2]

    results = [federated_epoch_close(c, pricing="equilibrated") for c in canonicals]

    # Strict equality on the authoritative payload — the part that
    # gets anchored on chain. Deterministic at byte level across
    # honest daemons.
    assert (
        results[0]["authoritative_payload"]
        == results[1]["authoritative_payload"]
        == results[2]["authoritative_payload"]
    )
    # And on individual top-level fields used by 5.6's chain
    # submission:
    assert results[0]["agent_mint"] == results[1]["agent_mint"] == results[2]["agent_mint"]
    assert results[0]["agent_novelty"] == results[1]["agent_novelty"] == results[2]["agent_novelty"]
    assert results[0]["epoch_root"] == results[1]["epoch_root"] == results[2]["epoch_root"]

    # node_mint uses ephemeral UUID node ids on the charter roots so
    # it is NOT cross-daemon-equal. It stays in the local diagnostic
    # surface; the chain anchor doesn't commit it.
    # (We don't assert anything about node_mint here.)

    # Both agents minted something positive.
    assert results[0]["agent_mint"].get("alice", 0) > 0
    assert results[0]["agent_mint"].get("bob", 0) > 0


def test_serialized_json_is_byte_identical_across_daemons():
    """The agent_mint dict, when JSON-serialized with sort_keys=False,
    must produce the same bytes on every daemon. This is the property
    a chain-anchor CID would commit to."""
    rpb = "rpb_fed_b"
    senders = [Keypair.generate() for _ in range(3)]
    chains = []
    for i, kp in enumerate(senders):
        chains.append(_make_chain(rpb, kp, [
            {"charter": (0.0, 0.0, 0.5, 0.0), "embed_idx": i * 10, "label": f"s{i}"},
            {"charter": (0.0, 0.0, 0.0, 0.5), "embed_idx": i * 10 + 1, "label": f"s{i}b"},
        ], agent_id=f"agent_{i}"))
    all_batches = [b for c in chains for b in c]

    rng = random.Random(11)
    deliveries = [list(all_batches) for _ in range(3)]
    for d in deliveries:
        rng.shuffle(d)

    serialized = []
    for d in deliveries:
        canonical = canonical_order(d)
        result = federated_epoch_close(canonical, pricing="equilibrated")
        # Serialize WITHOUT sort_keys — relies on the wrapper's
        # canonical insertion order.
        payload = json.dumps({
            "agent_mint": result["agent_mint"],
            "epoch_root": result["epoch_root"],
        }).encode("utf-8")
        serialized.append(payload)

    h0, h1, h2 = (hashlib.sha256(s).hexdigest() for s in serialized)
    assert h0 == h1 == h2, (h0, h1, h2)


# ---------------------------------------------------------------------------
# Honest + malformed mix
# ---------------------------------------------------------------------------


def test_malformed_sender_contributes_zero_others_unaffected():
    """One sender's chain has a gap (malformed). Three daemons all
    drop that sender via canonical_order, all run reconcile on the
    same survivor set, all produce the SAME agent_mint with the
    malformed sender at zero."""
    rpb = "rpb_fed_c"
    kp_honest = Keypair.generate()
    kp_bad = Keypair.generate()
    honest_chain = _make_chain(rpb, kp_honest, [
        {"charter": (0.6, 0.0, 0.0, 0.0), "embed_idx": 5, "label": "h1"},
        {"charter": (0.0, 0.0, 0.6, 0.0), "embed_idx": 6, "label": "h2"},
    ], agent_id="honest_alice")
    bad_chain = _make_chain(rpb, kp_bad, [
        {"charter": (0.0, 0.6, 0.0, 0.0), "embed_idx": 7, "label": "x1"},
        {"charter": (0.0, 0.0, 0.0, 0.6), "embed_idx": 8, "label": "x2"},
        {"charter": (0.0, 0.6, 0.6, 0.0), "embed_idx": 9, "label": "x3"},
    ], agent_id="malformed_eve")
    # Drop the middle batch from bad — gap.
    bad_with_gap = [bad_chain[0], bad_chain[2]]

    full = honest_chain + bad_with_gap
    rng = random.Random(99)
    deliveries = [list(full) for _ in range(3)]
    for d in deliveries:
        rng.shuffle(d)

    results = [
        federated_epoch_close(canonical_order(d), pricing="equilibrated")
        for d in deliveries
    ]

    # All three identical.
    assert results[0]["agent_mint"] == results[1]["agent_mint"] == results[2]["agent_mint"]

    # Honest alice minted something, malformed eve got nothing.
    mint = results[0]["agent_mint"]
    assert mint.get("honest_alice", 0) > 0
    assert "malformed_eve" not in mint or mint["malformed_eve"] == 0


def test_empty_canonical_yields_empty_authoritative_map():
    """No senders accepted -> empty agent_mint; epoch_root is the
    domain-separated empty marker."""
    canonical = canonical_order([])
    result = federated_epoch_close(canonical)
    assert result["agent_mint"] == {}
    assert result["agent_novelty"] == {}
    assert result["epoch_root"] == EMPTY_EPOCH_ROOT.hex()
    assert result["scope"] == "federated"
    assert result["authoritative"] is True


# ---------------------------------------------------------------------------
# Float-determinism stress
# ---------------------------------------------------------------------------


def test_many_agents_many_nodes_deterministic():
    """A larger reconciliation (many agents, many nodes) still
    produces dict-equal results across runs."""
    rpb = "rpb_fed_d"
    senders = [Keypair.generate() for _ in range(6)]
    chains = []
    for i, kp in enumerate(senders):
        chains.append(_make_chain(rpb, kp, [
            {"charter": (0.5 if i % 4 == j else 0.0
                         for j in range(4)),
             "embed_idx": (i * 5 + k),
             "label": f"a{i}_b{k}"}
            for k in range(3)
        ], agent_id=f"agent_{i:02d}"))
    # Note: the comprehension above is broken (charter is a generator);
    # rebuild explicitly.
    chains = []
    for i, kp in enumerate(senders):
        spec = []
        for k in range(3):
            charter = [0.0, 0.0, 0.0, 0.0]
            charter[i % 4] = 0.5
            spec.append({
                "charter": tuple(charter),
                "embed_idx": i * 5 + k,
                "label": f"a{i}_b{k}",
            })
        chains.append(_make_chain(rpb, kp, spec, agent_id=f"agent_{i:02d}"))
    all_batches = [b for c in chains for b in c]

    rng = random.Random(2026)
    deliveries = [list(all_batches) for _ in range(3)]
    for d in deliveries:
        rng.shuffle(d)

    results = [
        federated_epoch_close(canonical_order(d), pricing="equilibrated")
        for d in deliveries
    ]
    assert results[0]["agent_mint"] == results[1]["agent_mint"] == results[2]["agent_mint"]
    # And the totals match too (cheap secondary check).
    assert results[0]["total_mint"] == results[1]["total_mint"] == results[2]["total_mint"]


# ---------------------------------------------------------------------------
# Reconcile-only path (no replay) — for callers who already have a
# replayed world + snapshots and just want determinism.
# ---------------------------------------------------------------------------


def test_reconcile_wrapper_is_deterministic():
    """federated_reconcile_epoch (the inner wrapper) is deterministic
    given identical world + snapshots + events."""
    from world_model.generalized import equilibrate
    from nodes.common.world_model_substrate.adapter import build_charter_world
    from nodes.common.world_model_substrate.aggregate import apply_events
    from nodes.common.world_model_substrate.reconcile import EpochSnapshots

    rpb = "rpb_fed_e"
    kp = Keypair.generate()
    chain = _make_chain(rpb, kp, [
        {"charter": (0.5, 0.0, 0.0, 0.0), "embed_idx": 1, "label": "x"},
        {"charter": (0.0, 0.5, 0.0, 0.0), "embed_idx": 2, "label": "y"},
    ], agent_id="test_agent")

    def _run():
        world = build_charter_world(embedding_dim=1024)
        snaps = EpochSnapshots()
        snaps.record_start(world)
        events = []
        for b in chain:
            apply_events(world, b.events)
            equilibrate(world, max_rounds=8, tolerance=1e-3)
            events.extend(b.events)
        snaps.record_close(world)
        return federated_reconcile_epoch(world, snaps, events)

    r1 = _run()
    r2 = _run()
    # Agent-keyed dicts (deterministic across runs) must match.
    assert r1["agent_mint"] == r2["agent_mint"]
    assert r1["agent_novelty"] == r2["agent_novelty"]
    assert r1["total_mint"] == r2["total_mint"]
    assert r1["total_novelty"] == r2["total_novelty"]
    # node_mint uses ephemeral UUID node ids (charter root nodes get
    # fresh uuid.uuid4() in each build_charter_world call), so its
    # KEYS won't match across runs. Sum of values must, since the
    # same nodes minted by content even if their ids differ.
    assert sum(r1["node_mint"].values()) == sum(r2["node_mint"].values())
