"""Bit-identical federated closes across daemons.

Determinism contract: given the same canonical batch sequence, carry-over
and pinned chain inputs, every honest daemon's ``federated_epoch_close``
produces the SAME ``agent_mint`` dict at the byte level (same keys in the
same order, same values to the last bit) and the same authoritative
payload. That is what lets the anchor commit one attribution without
disputes.

Tested:

  1. Three daemons receive the same batches in different arrival orders;
     all produce identical payloads, mint maps and epoch roots.
  2. The serialized JSON of the mint map is byte-identical across daemons.
  3. A malformed (broken-chain) sender is dropped by canonical ordering:
     its registration never lands, so its tool mints nothing, and the
     others are unaffected.
  4. An empty epoch yields an empty mint map and the empty-epoch root.
  5. A larger library (many authors, many reviewers) is still identical.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List

from nodes.common.canonical_ordering import (
    EMPTY_EPOCH_ROOT,
    canonical_order,
)
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import federated_epoch_close

E18 = 10 ** 18
SUPPLY = 1000 * E18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plain_event(seq: int, agent_id: str) -> Dict[str, Any]:
    """An event the close ignores (only tool registrations matter)."""
    return {"kind": "observation_added", "seq": seq, "author_agent": agent_id,
            "obs_id": f"obs_{agent_id}_{seq}", "coords": [0.5, 0.1 * seq],
            "label": f"{agent_id} {seq}"}


def _tool_registration(author: str, digest: str) -> Dict[str, Any]:
    return {
        "kind": "tool_registered",
        "seq": 1,
        "author_agent": author,
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": "pinned", "author": author},
    }


def _append_batch(chain: List[EventBatch], rpb: str, kp: Keypair,
                  events: List[Dict[str, Any]]) -> List[EventBatch]:
    """Extend a sender's hash-linked chain with one more batch."""
    prev_hash = chain[-1].content_hash() if chain else b""
    b = EventBatch(
        rpb_address=rpb,
        sender_pubkey=kp.public_key,
        batch_seq=len(chain) + 1,
        events=events,
        prev_batch_hash=prev_hash,
        timestamp=1_700_000_100.0 + len(chain),
    )
    return chain + [b]


def _review_inputs(digests: List[str], reviewers: List[str]) -> Dict[str, Any]:
    """REP shares and a ToolBond snapshot in which every reviewer reviewed
    every tool in this window (accuracy 1, a positive correctness rating)."""
    share = 1.0 / (len(reviewers) + 1)
    snapshot = {
        "supply": SUPPLY,
        "tools": {d: {"axes": {"correctness": {"sum": 70 * E18, "mass": E18}},
                      "harm": {"sum": 0, "mass": 0}} for d in digests},
        "rounds": {d: [{"epoch": 1, "supply": SUPPLY, "prior_mass": {},
                        "reviews": [{"reviewer": r, "stake": 0, "accuracy": E18,
                                     "axes": ["correctness"]} for r in reviewers]}]
                   for d in digests},
    }
    return {"rep_shares": {r: share for r in reviewers},
            "review_snapshot": snapshot, "emission_pool": 10.0}


# ---------------------------------------------------------------------------
# Bit-identical determinism
# ---------------------------------------------------------------------------


def test_three_daemons_produce_byte_identical_maps():
    rpb = "rpb_fed_a"
    kp_alice, kp_bob = Keypair.generate(), Keypair.generate()
    dig_a, dig_b = "aa" * 32, "bb" * 32
    chain_a = _append_batch([], rpb, kp_alice, [_plain_event(1, "alice")])
    chain_a = _append_batch(chain_a, rpb, kp_alice, [_tool_registration("alice", dig_a)])
    chain_b = _append_batch([], rpb, kp_bob, [_tool_registration("bob", dig_b)])
    inputs = _review_inputs([dig_a, dig_b], ["0x" + "r1" * 20, "0x" + "r2" * 20])

    rng = random.Random(7)
    arrivals = [list(chain_a + chain_b) for _ in range(3)]
    for a in arrivals:
        rng.shuffle(a)
    canonicals = [canonical_order(a) for a in arrivals]
    seq_hashes = [[b.content_hash() for b in c.ordered_batches] for c in canonicals]
    assert seq_hashes[0] == seq_hashes[1] == seq_hashes[2]

    results = [federated_epoch_close(c, **inputs) for c in canonicals]
    assert (results[0]["authoritative_payload"]
            == results[1]["authoritative_payload"]
            == results[2]["authoritative_payload"])
    assert results[0]["agent_mint"] == results[1]["agent_mint"] == results[2]["agent_mint"]
    assert results[0]["epoch_root"] == results[1]["epoch_root"] == results[2]["epoch_root"]
    # Both tool authors minted (their reviews concluded this close).
    assert results[0]["agent_mint"].get("alice", 0) > 0
    assert results[0]["agent_mint"].get("bob", 0) > 0
    assert results[0]["authoritative_payload"]["schema"] == 7


def test_serialized_json_is_byte_identical_across_daemons():
    rpb = "rpb_fed_b"
    chains: List[EventBatch] = []
    digests = []
    for i in range(3):
        kp = Keypair.generate()
        d = f"{i + 1:02x}" * 32
        digests.append(d)
        c = _append_batch([], rpb, kp, [_plain_event(1, f"agent_{i}")])
        chains += _append_batch(c, rpb, kp, [_tool_registration(f"agent_{i}", d)])
    inputs = _review_inputs(digests, ["0x" + "r1" * 20])

    rng = random.Random(11)
    serialized = []
    for _ in range(3):
        d = list(chains)
        rng.shuffle(d)
        result = federated_epoch_close(canonical_order(d), **inputs)
        serialized.append(json.dumps({"agent_mint": result["agent_mint"],
                                      "epoch_root": result["epoch_root"]}).encode())
    h0, h1, h2 = (hashlib.sha256(s).hexdigest() for s in serialized)
    assert h0 == h1 == h2, (h0, h1, h2)


# ---------------------------------------------------------------------------
# Honest + malformed mix
# ---------------------------------------------------------------------------


def test_malformed_sender_contributes_zero_others_unaffected():
    """Eve's chain has a gap, so canonical ordering drops her: her tool's
    registration never lands and its concluding reviews pay nobody."""
    rpb = "rpb_fed_c"
    kp_honest, kp_bad = Keypair.generate(), Keypair.generate()
    dig_h, dig_e = "cc" * 32, "dd" * 32
    honest = _append_batch([], rpb, kp_honest, [_plain_event(1, "honest_alice")])
    honest = _append_batch(honest, rpb, kp_honest,
                           [_tool_registration("honest_alice", dig_h)])
    bad: List[EventBatch] = []
    for i in range(3):
        bad = _append_batch(bad, rpb, kp_bad, [_plain_event(i + 1, "malformed_eve")])
    bad = _append_batch(bad, rpb, kp_bad, [_tool_registration("malformed_eve", dig_e)])
    bad_with_gap = [bad[0], bad[2], bad[3]]
    inputs = _review_inputs([dig_h, dig_e], ["0x" + "r1" * 20])

    rng = random.Random(99)
    results = []
    for _ in range(3):
        d = list(honest + bad_with_gap)
        rng.shuffle(d)
        results.append(federated_epoch_close(canonical_order(d), **inputs))
    assert results[0]["agent_mint"] == results[1]["agent_mint"] == results[2]["agent_mint"]
    mint = results[0]["agent_mint"]
    assert mint.get("honest_alice", 0) > 0
    assert "malformed_eve" not in mint
    assert dig_e not in results[0]["tool_registrations"]


def test_empty_canonical_yields_empty_authoritative_map():
    result = federated_epoch_close(canonical_order([]))
    assert result["agent_mint"] == {}
    assert result["epoch_root"] == EMPTY_EPOCH_ROOT.hex()
    assert result["scope"] == "federated"
    assert result["authoritative"] is True
    assert set(result["authoritative_payload"]) == {
        "schema", "epoch_root", "agent_mint", "total_mint", "output_decimals",
        "charter_hash", "sanctioned_households", "n_batches", "n_events"}


# ---------------------------------------------------------------------------
# Float-determinism stress
# ---------------------------------------------------------------------------


def test_many_agents_many_tools_deterministic():
    rpb = "rpb_fed_d"
    chains: List[EventBatch] = []
    digests = []
    for i in range(6):
        kp = Keypair.generate()
        chain: List[EventBatch] = []
        for k in range(3):
            d = hashlib.sha256(f"tool-{i}-{k}".encode()).hexdigest()
            digests.append(d)
            chain = _append_batch(chain, rpb, kp, [
                _plain_event(k + 1, f"agent_{i:02d}"),
                _tool_registration(f"agent_{i:02d}", d)])
        chains += chain
    reviewers = ["0x" + f"{r:02x}" * 20 for r in range(1, 6)]
    inputs = _review_inputs(digests, reviewers)

    rng = random.Random(2026)
    results = []
    for _ in range(3):
        d = list(chains)
        rng.shuffle(d)
        results.append(federated_epoch_close(canonical_order(d), **inputs))
    assert results[0]["agent_mint"] == results[1]["agent_mint"] == results[2]["agent_mint"]
    assert results[0]["total_mint"] == results[1]["total_mint"] == results[2]["total_mint"]
    assert len(results[0]["tool_mint"]) == len(digests)


def test_household_mint_sums_per_owner_and_keeps_unbound_agents():
    from nodes.common.federated_reconcile import household_mint
    mint = {"0xAgentA": 3.0, "0xagentb": 2.0, "0xAgentC": 4.0}
    owners = {"0xagenta": "0xOWNER1", "0xAgentB": "0xowner1"}
    assert household_mint(mint, owners, 10) == {"0xagentc": 4.0, "0xowner1": 5.0}
    assert household_mint(mint, None, 10) == {
        "0xagenta": 3.0, "0xagentb": 2.0, "0xagentc": 4.0}
