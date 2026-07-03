"""Ledger pricing + violator-pays gate (docs/ledger_pricing.md).

Change 1 — pricing modes on federated_epoch_close:
  - default "ledger": causal events only, NO equilibrate rounds, NO
    derived-sprout capture. Per-node score = net_score tree recursion.
  - "equilibrated": the pre-phase8 kernel, unchanged.
  - Ledger determinism: net_score memoization over co-parented cycles
    is fixed to sorted node-id evaluation order — double-close on fresh
    worlds is byte-identical, including a co-parented-cycle case.

Change 2 — violator-pays gate:
  - per-(node, agent) attribution is scaled per node by
    (1 - gate_strength*violation) BEFORE aggregating per agent, so a
    flagged node's suppression falls only on its author. With the
    emission pool, the violator's absolute mint DROPS and an honest
    agent's absolute mint RISES.
  - zero violations => identical to no-gate.
  - the gated+pooled close is deterministic (double-run byte-identical).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import (
    federated_epoch_close,
    federated_reconcile_epoch,
)
from nodes.common.world_model_substrate.adapter import (
    N_DIMS,
    build_charter_world,
)
from nodes.common.world_model_substrate.aggregate import apply_events
from nodes.common.world_model_substrate.mint_gate import charter_violation_score
from nodes.common.world_model_substrate.reconcile import (
    EpochSnapshots,
    scale_node_agent_mint_by_violation,
    snapshot_node_scores_ordered,
)


EMBED = 8


# ---------------------------------------------------------------------------
# Event / batch construction
# ---------------------------------------------------------------------------


def _coords(axis_index: int, tail_idx: int, mag: float = 0.6):
    """Charter head (one-hot on axis_index) + embedding tail one-hot."""
    head = [1.0 if j == axis_index else 0.0 for j in range(N_DIMS)]
    tail = [0.0] * EMBED
    tail[tail_idx % EMBED] = mag
    return head + tail


def _sprout(
    seq: int,
    author: str,
    tendency_id: str,
    axis_index: int,
    tail_idx: int,
    node_label: str,
    *,
    parent_id: str,
    position: str = "pro",
    content: str = "",
) -> Dict[str, Any]:
    """A raw sub_claim_sprouted event dict with an authored unit post
    (so it moves score in ledger mode without equilibration)."""
    coords = _coords(axis_index, tail_idx)
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": author,
        "tendency_id": tendency_id,
        "parent_id": parent_id,
        "node_id": node_label,        # solver-side label; remapped on replay
        "position": position,
        "coords": coords,
        "polarity_axis": coords,
        "content": content or node_label,
        "author_post": True,
    }


def _batch(rpb: str, kp: Keypair, seq: int, prev: bytes,
           events: List[Dict[str, Any]]) -> EventBatch:
    return EventBatch(
        rpb_address=rpb,
        sender_pubkey=kp.public_key,
        batch_seq=seq,
        events=events,
        prev_batch_hash=prev,
        timestamp=1_700_000_000.0 + seq,
    )


def _chain(rpb: str, kp: Keypair,
           batches_events: List[List[Dict[str, Any]]]) -> List[EventBatch]:
    out: List[EventBatch] = []
    prev = b""
    for i, evs in enumerate(batches_events, start=1):
        b = _batch(rpb, kp, i, prev, evs)
        out.append(b)
        prev = b.content_hash()
    return out


# A CON post under a charter tendency whose content references a target
# node id — this is what charter_violation_score reads as a violation
# flag once it has positive standing.
def _charter_con(seq: int, author: str, charter_tendency: str,
                 axis_index: int, tail_idx: int, target_node_id: str,
                 node_label: str) -> Dict[str, Any]:
    coords = _coords(axis_index, tail_idx)
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": author,
        "tendency_id": charter_tendency,
        "parent_id": f"root_{charter_tendency}",
        "node_id": node_label,
        "position": "con",
        "coords": coords,
        "polarity_axis": coords,
        # content must contain the target node id so the gate matches it.
        "content": f"violates: {target_node_id}",
        "author_post": True,
    }


# ---------------------------------------------------------------------------
# Change 1 — ledger mode replays without equilibration
# ---------------------------------------------------------------------------


def test_ledger_mode_mints_from_net_score_no_equilibration():
    """A PRO sprout with an authored post under a charter root moves
    that root's net_score in ledger mode (no equilibrate needed) and
    the author mints."""
    rpb = "rpb_ledger_a"
    kp = Keypair.generate()
    chain = _chain(rpb, kp, [
        [_sprout(1, "alice", "correctness", 4, 0, "c1",
                 parent_id="root_correctness")],
        [_sprout(1, "alice", "simplicity", 5, 1, "s1",
                 parent_id="root_simplicity")],
    ])
    result = federated_epoch_close(canonical_order(chain), embedding_dim=EMBED)
    assert result["pricing"] == "ledger"
    assert result["agent_mint"].get("alice", 0.0) > 0.0


def test_ledger_default_and_equilibrated_are_distinct_paths():
    """Same canonical sequence, different pricing modes -> both run,
    default is ledger, and the mode is recorded in the payload."""
    rpb = "rpb_ledger_b"
    kp = Keypair.generate()
    chain = _chain(rpb, kp, [
        [_sprout(1, "alice", "correctness", 4, 0, "c1",
                 parent_id="root_correctness")],
    ])
    default = federated_epoch_close(canonical_order(chain), embedding_dim=EMBED)
    ledger = federated_epoch_close(
        canonical_order(chain), embedding_dim=EMBED, pricing="ledger")
    equil = federated_epoch_close(
        canonical_order(chain), embedding_dim=EMBED, pricing="equilibrated")
    assert default["pricing"] == "ledger"
    assert default["agent_mint"] == ledger["agent_mint"]
    assert default["authoritative_payload"]["pricing"] == "ledger"
    assert equil["pricing"] == "equilibrated"


def test_bad_pricing_mode_rejected():
    kp = Keypair.generate()
    chain = _chain("rpb_x", kp, [
        [_sprout(1, "a", "correctness", 4, 0, "c1",
                 parent_id="root_correctness")],
    ])
    with pytest.raises(ValueError, match="pricing"):
        federated_epoch_close(
            canonical_order(chain), embedding_dim=EMBED, pricing="bogus")


# ---------------------------------------------------------------------------
# Change 1 — cycle-order determinism
# ---------------------------------------------------------------------------


def _cycle_chain(rpb: str, kp: Keypair) -> List[EventBatch]:
    """Build a canonical sequence that produces a co-parented cycle.

    A (under root_correctness, PRO) -> B (under A, PRO) -> then re-sprout
    A's coordinates as a PRO child of B. Because node ids are
    content-addressed by (anchor, axis), the third sprout resolves to
    the SAME node A and only appends a parent edge (A becomes a child of
    B while B is already a descendant of A) — a genuine cycle.
    """
    # All three sprouts ride ONE batch so the intra-batch solver-id
    # remap resolves "A" and "B" to their live nodes (cross-batch,
    # solver labels reset and would fall back to the root).
    a = _sprout(1, "alice", "correctness", 4, 0, "A",
                parent_id="root_correctness")
    b = _sprout(2, "alice", "correctness", 4, 1, "B", parent_id="A")
    # third sprout: same coords as A (axis 4, tail 0) but parented under B.
    a_again = _sprout(3, "alice", "correctness", 4, 0, "A2", parent_id="B")
    return _chain(rpb, kp, [[a, b, a_again]])


def test_ledger_double_close_byte_identical_plain():
    """Fresh worlds, same canonical sequence, ledger mode: the agent_mint
    map is byte-identical across two independent closes."""
    rpb = "rpb_ledger_det"
    kp = Keypair.generate()
    chain = _chain(rpb, kp, [
        [_sprout(1, "alice", "correctness", 4, 0, "c1",
                 parent_id="root_correctness")],
        [_sprout(1, "bob", "simplicity", 5, 2, "s1",
                 parent_id="root_simplicity")],
    ])
    r1 = federated_epoch_close(canonical_order(chain), embedding_dim=EMBED)
    r2 = federated_epoch_close(canonical_order(chain), embedding_dim=EMBED)
    assert json.dumps(r1["agent_mint"]) == json.dumps(r2["agent_mint"])
    assert r1["authoritative_payload"] == r2["authoritative_payload"]


def test_ledger_coparented_cycle_double_close_byte_identical():
    """The load-bearing determinism proof: a co-parented CYCLE, closed
    twice on fresh worlds, yields byte-identical agent_mint AND
    node-level scores. net_score memoization over the cycle is
    eval-order sensitive; sorted-id evaluation makes it reproducible."""
    rpb = "rpb_ledger_cycle"
    kp = Keypair.generate()
    chain = _cycle_chain(rpb, kp)

    # First prove the world actually contains a cycle: A in B.children
    # and B in A.children.
    world = build_charter_world(embedding_dim=EMBED)
    for b in chain:
        apply_events(world, b.events, equilibrate_after=False)
    tend = world.tendencies["correctness"]
    a_id = tend._content_address(  # type: ignore[attr-defined]
        tuple(_coords(4, 0)), tuple(_coords(4, 0)))
    b_id = tend._content_address(  # type: ignore[attr-defined]
        tuple(_coords(4, 1)), tuple(_coords(4, 1)))
    node_a = tend.tree.get_node(a_id)
    node_b = tend.tree.get_node(b_id)
    assert node_a is not None and node_b is not None
    child_ids_a = {c.id for c in node_a.pro_children + node_a.con_children}
    child_ids_b = {c.id for c in node_b.pro_children + node_b.con_children}
    assert b_id in child_ids_a and a_id in child_ids_b, "cycle not formed"

    # Ordered snapshot must not blow the stack and must be deterministic.
    s1 = snapshot_node_scores_ordered(world)
    s2 = snapshot_node_scores_ordered(world)
    assert s1 == s2

    # Now the end-to-end double close on fresh worlds.
    r1 = federated_epoch_close(canonical_order(chain), embedding_dim=EMBED)
    r2 = federated_epoch_close(canonical_order(chain), embedding_dim=EMBED)
    assert json.dumps(r1["agent_mint"]) == json.dumps(r2["agent_mint"])
    assert r1["authoritative_payload"] == r2["authoritative_payload"]


# ---------------------------------------------------------------------------
# Change 2 — violator-pays gate
# ---------------------------------------------------------------------------


def _two_author_events() -> Tuple[List[Dict[str, Any]], str]:
    """Two authors each sprout one PRO node under a charter root.
    Returns (events, honest_node_id). violator == 'eve', honest ==
    'alice'. eve's node is the one we'll flag."""
    # honest alice under correctness
    alice = _sprout(1, "alice", "correctness", 4, 0, "alice_node",
                    parent_id="root_correctness")
    # violator eve under simplicity
    eve = _sprout(1, "eve", "simplicity", 5, 1, "eve_node",
                  parent_id="root_simplicity")
    return [alice, eve], "alice"


def _reconcile_over(events, *, apply_gate, emission_pool=None):
    """Replay events onto a fresh world (ledger-style, no equilibrate),
    snapshot, and run federated_reconcile_epoch. Mirrors the ledger
    path's node_id remap so attribution reaches the real authors."""
    world = build_charter_world(embedding_dim=EMBED)
    snaps = EpochSnapshots()
    snaps.start = snapshot_node_scores_ordered(world)
    remap: Dict[str, str] = {}
    apply_events(world, events, equilibrate_after=False, remap_out=remap)
    snaps.close = snapshot_node_scores_ordered(world)
    attrib_events = []
    for ev in events:
        if ev.get("kind") == "sub_claim_sprouted":
            live = remap.get(ev.get("node_id", ""))
            if live and live != ev.get("node_id"):
                ev = dict(ev)
                ev["node_id"] = live
        attrib_events.append(ev)
    return federated_reconcile_epoch(
        world, snaps, attrib_events,
        apply_gate=apply_gate,
        emission_pool=emission_pool,
    )


def test_violator_pays_honest_rises_violator_drops_after_pool():
    """Fixed pool. One flagged node (authored by the violator). After
    the gate + pool: violator's absolute mint DROPS vs no-gate, honest
    agent's absolute mint RISES vs no-gate."""
    events, _ = _two_author_events()

    # Need the flag to reference eve's live node id, which is content-
    # addressed. Compute it, then append a charter CON that wins standing.
    world = build_charter_world(embedding_dim=EMBED)
    apply_events(world, events, equilibrate_after=False)
    tend = world.tendencies["simplicity"]
    eve_node_id = tend._content_address(  # type: ignore[attr-defined]
        tuple(_coords(5, 1)), tuple(_coords(5, 1)))

    # A charter CON (under 'correctness') flagging eve's node, with an
    # authored post so it has positive net_score => real violation.
    con = _charter_con(2, "watchdog", "correctness", 4, 3,
                       eve_node_id, "flag_eve")
    flagged_events = events + [con]

    POOL = 100.0

    no_gate = _reconcile_over(flagged_events, apply_gate=False,
                              emission_pool=POOL)
    gated = _reconcile_over(flagged_events, apply_gate=True,
                            emission_pool=POOL)

    # Sanity: the gate saw a real violation on eve's node.
    world2 = build_charter_world(embedding_dim=EMBED)
    apply_events(world2, flagged_events, equilibrate_after=False)
    v = charter_violation_score(world2, eve_node_id)
    assert v > 0.0, "test setup failed to produce a real violation flag"

    honest_before = no_gate["agent_mint"].get("alice", 0.0)
    honest_after = gated["agent_mint"].get("alice", 0.0)
    violator_before = no_gate["agent_mint"].get("eve", 0.0)
    violator_after = gated["agent_mint"].get("eve", 0.0)

    assert violator_before > 0.0 and honest_before > 0.0
    # Violator strictly down, honest strictly up (absolute, post-pool).
    assert violator_after < violator_before
    assert honest_after > honest_before
    # Pool is conserved (both close to POOL; equal since raw_total>0).
    assert gated["total_mint"] == pytest.approx(POOL)
    assert no_gate["total_mint"] == pytest.approx(POOL)


def test_zero_violations_identical_to_no_gate():
    """No charter CON => no violation => gated result equals no-gate
    result exactly (before and after the pool)."""
    events, _ = _two_author_events()
    POOL = 50.0
    no_gate = _reconcile_over(events, apply_gate=False, emission_pool=POOL)
    gated = _reconcile_over(events, apply_gate=True, emission_pool=POOL)
    assert gated["agent_mint"] == no_gate["agent_mint"]
    assert gated["total_mint"] == pytest.approx(no_gate["total_mint"])


def test_gated_pooled_close_deterministic():
    """The full gated + pooled federated close is byte-identical on a
    double run (fresh worlds)."""
    rpb = "rpb_gate_det"
    kp = Keypair.generate()
    events, _ = _two_author_events()
    # Reuse the same two events across two batches from one signer.
    chain = _chain(rpb, kp, [[events[0]], [events[1]]])

    def _run():
        return federated_epoch_close(
            canonical_order(chain), embedding_dim=EMBED,
            apply_gate=True, emission_pool=100.0,
        )

    r1 = _run()
    r2 = _run()
    assert json.dumps(r1["agent_mint"]) == json.dumps(r2["agent_mint"])
    assert r1["authoritative_payload"] == r2["authoritative_payload"]


def test_scale_helper_only_hits_flagged_nodes_author():
    """Unit-level: scaling a flagged node's per-agent mint leaves an
    honest node's author untouched before aggregation."""
    node_agent_mint = {
        "n_honest": {"alice": 4.0},
        "n_bad": {"eve": 4.0},
    }
    node_violation = {"n_honest": 0.0, "n_bad": 1.0}
    out = scale_node_agent_mint_by_violation(
        node_agent_mint, node_violation, gate_strength=1.0)
    assert out["alice"] == pytest.approx(4.0)   # untouched
    assert out.get("eve", 0.0) == pytest.approx(0.0)   # fully suppressed
