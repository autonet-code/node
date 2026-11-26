"""Bit-identical federated closes across daemons.

Determinism contract: given the same canonical batch sequence, carry-over
and pinned chain inputs, every honest daemon's ``federated_epoch_close``
produces the SAME ``claims`` at the byte level (same keys in the
same order, same values to the last bit) and the same authoritative
payload. That is what lets the anchor commit one attribution without
disputes.

Tested:

  1. Three daemons receive the same batches in different arrival orders;
     all produce identical payloads, claims and epoch roots.
  2. The serialized JSON of the claims is byte-identical across daemons.
  3. A malformed (broken-chain) sender is dropped by canonical ordering:
     its registration never lands, so its tool earns nothing, and the
     others are unaffected.
  4. An empty epoch yields no claims and the empty-epoch root.
  5. A larger library (many authors, many reviewers) is still identical.
  6. The basket split: two tokens of different decimals, each pool
     conserved (paid + carry), sanctions withheld into the carry, the
     value of each claim at the pinned parity.
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
USDC = "0x" + "a6" * 20          # 6 decimals, parity 1e30
DAI = "0x" + "1b" * 20           # 18 decimals, parity 1e18
WINDOW = {USDC: 10 * 10 ** 6, DAI: 10 * E18}
PARITY = {USDC: 10 ** 30, DAI: 10 ** 18}


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
    from tests.test_tool_mint import tool_cells
    snapshot = {
        "supply": SUPPLY,
        "tools": {d: tool_cells(correctness=0.7) for d in digests},
        "rounds": {d: [{"epoch": 1, "supply": SUPPLY,
                        "reviews": [{"reviewer": r, "accuracy": E18}
                                    for r in reviewers]}]
                   for d in digests},
    }
    return {"rep_shares": {r: share for r in reviewers},
            "review_snapshot": snapshot, "tool_share_window": dict(WINDOW),
            "parity": dict(PARITY)}


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
    assert results[0]["claims"] == results[1]["claims"] == results[2]["claims"]
    assert results[0]["epoch_root"] == results[1]["epoch_root"] == results[2]["epoch_root"]
    # Both tool authors earn (their reviews concluded this close).
    assert results[0]["claims"]["alice"]["value"] > 0
    assert results[0]["claims"]["bob"]["value"] > 0
    assert results[0]["authoritative_payload"]["schema"] == 10


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
        serialized.append(json.dumps({"claims": result["claims"],
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
    assert results[0]["claims"] == results[1]["claims"] == results[2]["claims"]
    mint = results[0]["claims"]
    assert mint["honest_alice"]["value"] > 0
    assert "malformed_eve" not in mint
    assert dig_e not in results[0]["tool_registrations"]


def test_empty_canonical_yields_empty_authoritative_map():
    result = federated_epoch_close(canonical_order([]))
    assert result["claims"] == {}
    assert result["epoch_root"] == EMPTY_EPOCH_ROOT.hex()
    assert result["scope"] == "federated"
    assert result["authoritative"] is True
    assert set(result["authoritative_payload"]) == {
        "schema", "epoch_root", "claims", "pools", "parity", "total_value",
        "rep_floor", "total_seed", "output_decimals",
        "charter_hash", "sanctioned_households", "concluded", "n_batches",
        "n_events"}


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
    assert results[0]["claims"] == results[1]["claims"] == results[2]["claims"]
    assert results[0]["total_value"] == results[1]["total_value"] == results[2]["total_value"]
    assert len(results[0]["tool_mint"]) == len(digests)


# ---------------------------------------------------------------------------
# The basket split
# ---------------------------------------------------------------------------


def _one_tool_close(**kw):
    """One tool by ``alice``, reviewed by two equal reviewers, over a
    two-token window."""
    rpb = "rpb_basket"
    kp = Keypair.generate()
    dig = "ee" * 32
    chain = _append_batch([], rpb, kp, [_tool_registration("alice", dig)])
    r1, r2 = "0x" + "11" * 20, "0x" + "22" * 20
    inputs = _review_inputs([dig], [r1, r2])
    inputs.update(kw)
    return federated_epoch_close(canonical_order(chain), **inputs), (r1, r2)


def _paid(result):
    out: Dict[str, int] = {}
    for c in result["claims"].values():
        for t, v in c["tokens"].items():
            out[t] = out.get(t, 0) + v
    return out


def test_basket_two_tokens_of_different_decimals():
    result, (r1, r2) = _one_tool_close()
    claims = result["claims"]
    assert set(claims) == {"alice", r1, r2}
    # Every earner takes the same fraction of each pool (to the floor).
    for c in claims.values():
        f6 = c["tokens"][USDC] / WINDOW[USDC]
        f18 = c["tokens"][DAI] / WINDOW[DAI]
        assert abs(f6 - f18) < 1e-6
        assert all(isinstance(v, int) for v in c["tokens"].values())
    # Equal reviewers take equal amounts.
    assert claims[r1]["tokens"] == claims[r2]["tokens"]
    assert result["pools"] == WINDOW and result["parity"] == PARITY


def test_basket_conserves_each_pool_exactly():
    result, _ = _one_tool_close()
    paid = _paid(result)
    carry = result["tool_rounds"]["pool_carry"]
    for t, pool in WINDOW.items():
        assert paid.get(t, 0) + carry.get(t, 0) == pool
    # The rest the author side does not pay (score below 1) is carried.
    assert carry[USDC] > 0 and carry[DAI] > 0


def test_the_carry_joins_the_next_pool_and_an_int_carry_reads_as_empty():
    first, _ = _one_tool_close()
    carry = first["tool_rounds"]["pool_carry"]
    quiet = federated_epoch_close(canonical_order([]), tool_rounds=first["tool_rounds"],
                                  tool_share_window={USDC: 5}, parity=PARITY)
    assert quiet["pools"] == {DAI: carry[DAI], USDC: carry[USDC] + 5}
    assert quiet["tool_rounds"]["pool_carry"] == quiet["pools"]
    old = federated_epoch_close(canonical_order([]), tool_rounds={"pool_carry": 123},
                                tool_share_window={USDC: 5})
    assert old["pools"] == {USDC: 5}


def test_sanctioned_household_is_withheld_into_the_carry():
    """The author's household is sanctioned: its share is split with the
    rest, then withheld, so it stays in the carry and nobody else gains."""
    clean, (r1, r2) = _one_tool_close()
    house = "0x" + "99" * 20
    sanctioned, _ = _one_tool_close(agent_owner_map={"alice": house},
                                    sanctioned_households=[house])
    assert house not in sanctioned["claims"] and "alice" not in sanctioned["claims"]
    assert sanctioned["claims"][r1] == clean["claims"][r1]
    assert sanctioned["claims"][r2] == clean["claims"][r2]
    withheld = sanctioned["withheld"]["alice"]
    assert withheld == clean["claims"]["alice"]["tokens"]
    carry = sanctioned["tool_rounds"]["pool_carry"]
    for t in WINDOW:
        assert carry[t] == clean["tool_rounds"]["pool_carry"][t] + withheld[t]
        assert _paid(sanctioned)[t] + carry[t] == WINDOW[t]


def test_values_are_each_token_at_its_parity_summed():
    result, _ = _one_tool_close()
    total = 0
    for c in result["claims"].values():
        expect = (c["tokens"][USDC] * PARITY[USDC] // 10 ** 30
                  + c["tokens"][DAI] * PARITY[DAI] // 10 ** 30)
        assert c["value"] == expect
        total += expect
    assert result["total_value"] == total
    assert result["authoritative_payload"]["total_value"] == total
    # A token out of the index (no parity) is still paid, at value 0.
    no_dai, _ = _one_tool_close(parity={USDC: 10 ** 30})
    for h, c in no_dai["claims"].items():
        assert c["tokens"][DAI] == result["claims"][h]["tokens"][DAI]
        assert c["value"] == c["tokens"][USDC]
    assert no_dai["parity"] == {DAI: 0, USDC: 10 ** 30}


def _two_author_close(**kw):
    rpb = "rpb_basket2"
    ka, kb = Keypair.generate(), Keypair.generate()
    da, db = "e1" * 32, "e2" * 32
    chain = (_append_batch([], rpb, ka, [_tool_registration("alice", da)])
             + _append_batch([], rpb, kb, [_tool_registration("bob", db)]))
    inputs = _review_inputs([da, db], ["0x" + "11" * 20])
    inputs.update(kw)
    return federated_epoch_close(canonical_order(chain), **inputs)


def test_household_amounts_are_added_per_owner_never_resplit():
    owner = "0x" + "77" * 20
    alone = _two_author_close()
    together = _two_author_close(agent_owner_map={"alice": owner, "bob": owner})
    assert "alice" not in together["claims"] and "bob" not in together["claims"]
    for t in WINDOW:
        assert together["claims"][owner]["tokens"][t] == (
            alone["claims"]["alice"]["tokens"][t] + alone["claims"]["bob"]["tokens"][t])
    assert together["claims_by_agent"]["alice"] == alone["claims"]["alice"]["tokens"]


def test_no_chain_no_split():
    result, (r1, _) = _one_tool_close(tool_share_window=None, parity=None)
    assert result["claims"] == {} and result["pools"] == {}
    assert result["tool_rounds"]["pool_carry"] == {}
    assert result["household_weights"][r1] > 0
