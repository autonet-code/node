"""Tool mint at the federated epoch close (docs/tool_substrate.md).

Author attribution ∝ effective standing × log1p(ok usage), anchored on
the manifest's claim node, merged through the violator-pays gate path.
Covers: end-to-end close with registration + receipts, bit-identical
double close, shuffled-delivery determinism, attested decay through
tool_receipt_history, carry-over registrations, and the no-tool-events
null case.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import (
    compute_tool_mint,
    federated_epoch_close,
)
from nodes.common.world_model_substrate.adapter import N_DIMS, build_charter_world
from nodes.common.world_model_substrate.aggregate import apply_events

EMBED_DIM = 1024
DIGEST = "ab" * 32
AUTHOR = "toolsmith"


def _coords(axis: int = 4, mag: float = 0.8) -> List[float]:
    out = [0.0] * (N_DIMS + EMBED_DIM)
    out[axis] = mag
    out[N_DIMS + 3] = 0.5  # some embedding-tail mass
    return out


def _registration_event(seq: int = 1, *, trust_class: str = "pinned",
                        digest: str = DIGEST) -> Dict[str, Any]:
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": AUTHOR,
        "tendency_id": "correctness",
        "parent_id": "solver_root",       # remaps to the live root
        "node_id": f"tm_{digest[:12]}",
        "position": "pro",
        "coords": _coords(),
        "polarity_axis": _coords(),
        "content": "tool echo_tool: echoes its input back",
        "author_post": True,              # immediate unit standing
        "artifact_digest": digest,
        "manifest_meta": {"trust_class": trust_class, "author": AUTHOR},
    }


def _receipt_event(seq: int, caller: str, *, ok: bool = True,
                   digest: str = DIGEST) -> Dict[str, Any]:
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": caller,
        "manifest_digest": digest,
        "tool_author": AUTHOR,
        "receipt_digest": f"r{seq:02d}" * 8,
        "ok": ok,
        "fee_atn": 0.0,
    }


def _batches(events: List[Dict[str, Any]], keypair: Keypair) -> List[EventBatch]:
    chain: List[EventBatch] = []
    prev_hash = b""
    for i, ev in enumerate(events, start=1):
        b = EventBatch(
            rpb_address="rpb_tooltest",
            sender_pubkey=keypair.public_key,
            batch_seq=i,
            events=[ev],
            prev_batch_hash=prev_hash,
            timestamp=1_700_000_000.0 + i,
        )
        chain.append(b)
        prev_hash = b.content_hash()
    return chain


def _standard_batches() -> List[EventBatch]:
    kp = Keypair.generate()
    return _batches([
        _registration_event(1),
        _receipt_event(2, "caller-1"),
        _receipt_event(3, "caller-2"),
        _receipt_event(4, "caller-1", ok=False),   # failure: no mint credit
    ], kp)


class TestToolMintClose:
    def test_close_mints_to_author(self):
        result = federated_epoch_close(canonical_order(_standard_batches()))
        tm = result["tool_mint"]
        assert DIGEST in tm
        entry = tm[DIGEST]
        assert entry["author"] == AUTHOR
        assert entry["ok_count"] == 2                       # failure excluded
        assert entry["standing"] > 0                        # author_post landed
        assert entry["mint"] > 0
        assert entry["mint"] == entry["standing"] * math.log1p(2)
        # Merged into the consensus attribution map.
        assert result["agent_mint"].get(AUTHOR, 0) >= round(entry["mint"], 6) * 0.5
        assert result["tool_registrations"][DIGEST]["author"] == AUTHOR

    def test_double_close_bit_identical(self):
        batches = _standard_batches()
        r1 = federated_epoch_close(canonical_order(list(batches)))
        r2 = federated_epoch_close(canonical_order(list(batches)))
        for key in ("agent_mint", "tool_mint",
                    "tool_registrations", "authoritative_payload"):
            assert json.dumps(r1[key]) == json.dumps(r2[key]), key

    def test_shuffled_delivery_deterministic(self):
        kp1, kp2 = Keypair.generate(), Keypair.generate()
        chain1 = _batches([_registration_event(1), _receipt_event(2, "c1")], kp1)
        chain2 = _batches([_receipt_event(1, "c2"), _receipt_event(2, "c3")], kp2)
        all_batches = chain1 + chain2
        rng = random.Random(11)
        results = []
        for _ in range(3):
            delivery = list(all_batches)
            rng.shuffle(delivery)
            results.append(federated_epoch_close(canonical_order(delivery)))
        assert results[0]["tool_mint"] == results[1]["tool_mint"] == results[2]["tool_mint"]
        assert results[0]["agent_mint"] == results[1]["agent_mint"] == results[2]["agent_mint"]

    def test_attested_mints_nothing(self):
        """Pinned-only emission (spec v2): connector-backed tools are
        publishable and debatable but draw nothing from the pool."""
        kp = Keypair.generate()
        batches = _batches([
            _registration_event(1, trust_class="attested"),
            _receipt_event(2, "caller-1"),
        ], kp)
        result = federated_epoch_close(canonical_order(batches))
        assert result["tool_mint"] == {}
        # ...but the registration still carries over (attribution is
        # permanent even for unminted classes).
        assert result["tool_registrations"][DIGEST]["trust_class"] == "attested"

    def test_no_tool_events_null_case(self):
        kp = Keypair.generate()
        batches = _batches([{
            "kind": "observation_added", "seq": 1, "author_agent": "alice",
            "obs_id": "obs_x", "coords": _coords(), "label": "plain work",
        }], kp)
        result = federated_epoch_close(canonical_order(batches))
        assert result["tool_mint"] == {}
        assert result["tool_registrations"] == {}


class TestComputeToolMint:
    def _world_with_registration(self):
        world = build_charter_world(embedding_dim=EMBED_DIM)
        apply_events(world, [_registration_event(1)], equilibrate_after=False)
        return world

    def test_carried_registration_attributes(self):
        """Receipts-only epoch: the carried registration map still
        attributes, provided the claim node exists in the (seeded)
        world."""
        world = self._world_with_registration()
        events = [_receipt_event(1, "caller-9")]
        out = compute_tool_mint(
            world, events,
            registrations={DIGEST: {"trust_class": "pinned", "author": AUTHOR}},
        )
        assert out["per_digest"][DIGEST]["author"] == AUTHOR
        assert out["per_digest"][DIGEST]["mint"] > 0
        assert out["registrations_next"][DIGEST]["author"] == AUTHOR

    def test_carried_registration_wins_over_reregistration(self):
        world = self._world_with_registration()
        rereg = _registration_event(1)
        rereg["manifest_meta"] = {"trust_class": "pinned", "author": "mallory"}
        events = [rereg, _receipt_event(2, "caller-9")]
        out = compute_tool_mint(
            world, events,
            registrations={DIGEST: {"trust_class": "pinned", "author": AUTHOR}},
        )
        assert out["per_digest"][DIGEST]["author"] == AUTHOR

    def test_usage_without_any_registration_mints_nothing(self):
        world = build_charter_world(embedding_dim=EMBED_DIM)
        out = compute_tool_mint(world, [_receipt_event(1, "caller-1")])
        assert out["per_digest"] == {}
        assert out["registrations_next"] == {}
