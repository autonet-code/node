"""Household voice at the federated close (balance-weighted voice).

Spec: docs/tool_substrate.md, Decision 2026-07-08 addendum. Callers
collapse to HOUSEHOLDS (proven owner wallet, agent-id fallback) before
log1p damping — N co-owned agents are one voice — and each household's
damped usage/review credit scales by its voice weight
(epsilon + household_ATN/supply, linear in balance). Covers: household
collapse, agent-splitting invariance, weight scaling on mint and
position drift, the epsilon floor for unknown households, the
weights=None legacy path, and close determinism with weights.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import pytest

from nodes.common.canonical_ordering import canonical_order
from nodes.common.event_gossip import EventBatch, Keypair
from nodes.common.federated_reconcile import (
    VOICE_EPSILON,
    federated_epoch_close,
)
from nodes.common.world_model_substrate.adapter import N_DIMS

EMBED_DIM = 1024
DIGEST = "cd" * 32
AUTHOR = "toolsmith"


def _coords(axis: int = 4, mag: float = 0.8) -> List[float]:
    out = [0.0] * (N_DIMS + EMBED_DIM)
    out[axis] = mag
    out[N_DIMS + 3] = 0.5
    return out


def _registration_event(seq: int = 1) -> Dict[str, Any]:
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
        "content": f"tool {DIGEST[:8]}: does a thing",
        "author_post": True,
        "artifact_digest": DIGEST,
        "manifest_meta": {"trust_class": "pinned", "author": AUTHOR},
    }


def _receipt_event(seq: int, caller: str, *,
                   axes: Dict[str, float] | None = None) -> Dict[str, Any]:
    ev = {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": caller,
        "manifest_digest": DIGEST,
        "tool_author": AUTHOR,
        "receipt_digest": f"r{seq:02d}" * 8,
        "ok": True,
        "fee_atn": 0.0,
        "attested": True,
        "score": 0.8,
    }
    if axes:
        ev["axes"] = dict(axes)
    return ev


def _vet_event(seq: int, vetter: str) -> Dict[str, Any]:
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": vetter,
        "manifest_digest": DIGEST,
        "tool_author": AUTHOR,
        "receipt_digest": f"v{seq:02d}" * 8,
        "ok": True,
        "fee_atn": 0.0,
        "vet": True,
    }


def _batches(events: List[Dict[str, Any]], keypair: Keypair) -> List[EventBatch]:
    chain: List[EventBatch] = []
    prev_hash = b""
    for i, ev in enumerate(events, start=1):
        b = EventBatch(
            rpb_address="rpb_voicetest",
            sender_pubkey=keypair.public_key,
            batch_seq=i,
            events=[ev],
            prev_batch_hash=prev_hash,
            timestamp=1_700_000_000.0 + i,
        )
        chain.append(b)
        prev_hash = b.content_hash()
    return chain


def _scenario(receipts: List[Dict[str, Any]]) -> List[EventBatch]:
    """Registration + vets (greenlight) from distinct keys, plus the
    given receipts on a third key."""
    out = _batches([_registration_event(1)], Keypair.generate())
    out += _batches([_vet_event(1, "vetter-1")], Keypair.generate())
    out += _batches([_vet_event(1, "vetter-2")], Keypair.generate())
    out += _batches(receipts, Keypair.generate())
    return out


class TestHouseholdCollapse:
    def test_same_owner_callers_are_one_voice(self):
        """Two agents under one owner = log1p(2), not 2*log1p(1) —
        the per-agent log1p amplification is closed."""
        batches = _scenario([
            _receipt_event(1, "agent-a"),
            _receipt_event(2, "agent-b"),
        ])
        owner_map = {"agent-a": "0xhouse", "agent-b": "0xhouse"}
        result = federated_epoch_close(
            canonical_order(batches), agent_owner_map=owner_map)
        entry = result["tool_mint"][DIGEST]
        assert entry["attesters"] == 1
        assert entry["mint"] == pytest.approx(math.log1p(2))

    def test_distinct_owners_stay_distinct_voices(self):
        batches = _scenario([
            _receipt_event(1, "agent-a"),
            _receipt_event(2, "agent-b"),
        ])
        owner_map = {"agent-a": "0xhouse1", "agent-b": "0xhouse2"}
        result = federated_epoch_close(
            canonical_order(batches), agent_owner_map=owner_map)
        entry = result["tool_mint"][DIGEST]
        assert entry["attesters"] == 2
        assert entry["mint"] == pytest.approx(2 * math.log1p(1))

    def test_agent_splitting_invariance(self):
        """One owner, same total attestations: 1 agent x4 == 4 agents
        x1. Registering more agents never gains weight."""
        one_agent = _scenario([
            _receipt_event(i, "agent-a") for i in range(1, 5)
        ])
        four_agents = _scenario([
            _receipt_event(i, f"agent-{c}")
            for i, c in enumerate("abcd", start=1)
        ])
        om_one = {"agent-a": "0xhouse"}
        om_four = {f"agent-{c}": "0xhouse" for c in "abcd"}
        r_one = federated_epoch_close(
            canonical_order(one_agent), agent_owner_map=om_one)
        r_four = federated_epoch_close(
            canonical_order(four_agents), agent_owner_map=om_four)
        assert (r_one["tool_mint"][DIGEST]["mint"]
                == pytest.approx(r_four["tool_mint"][DIGEST]["mint"]))
        assert r_one["tool_mint"][DIGEST]["mint"] == pytest.approx(
            math.log1p(4))

    def test_author_household_excluded(self):
        """A caller under the author's own owner wallet is the author's
        household — excluded (subsumes self-attestation)."""
        batches = _scenario([_receipt_event(1, "family-agent")])
        owner_map = {AUTHOR: "0xauthorhouse",
                     "family-agent": "0xauthorhouse"}
        result = federated_epoch_close(
            canonical_order(batches), agent_owner_map=owner_map)
        assert result["tool_mint"] == {}


class TestVoiceWeights:
    def test_weights_scale_mint_linearly(self):
        batches = _scenario([
            _receipt_event(1, "agent-a"),
            _receipt_event(2, "agent-b"),
        ])
        owner_map = {"agent-a": "0xrich", "agent-b": "0xpoor"}
        weights = {"0xrich": 0.55, "0xpoor": 0.05}
        result = federated_epoch_close(
            canonical_order(batches), agent_owner_map=owner_map,
            voice_weights=weights)
        entry = result["tool_mint"][DIGEST]
        assert entry["mint"] == pytest.approx(
            math.log1p(1) * 0.55 + math.log1p(1) * 0.05)

    def test_unknown_household_gets_epsilon_floor(self):
        """A weights map is present but doesn't know this household →
        the epsilon floor, not full weight. Zero-balance throwaway
        identities carry at most epsilon each."""
        batches = _scenario([_receipt_event(1, "agent-a")])
        result = federated_epoch_close(
            canonical_order(batches),
            agent_owner_map={"agent-a": "0xstranger"},
            voice_weights={"0xelse": 1.0})
        entry = result["tool_mint"][DIGEST]
        assert entry["mint"] == pytest.approx(
            math.log1p(1) * VOICE_EPSILON)

    def test_no_weights_means_unit_voice(self):
        """voice_weights=None (no chain access) = every household weighs
        1.0 — the pre-voice close output."""
        batches = _scenario([_receipt_event(1, "agent-a")])
        r_none = federated_epoch_close(canonical_order(list(batches)))
        assert r_none["tool_mint"][DIGEST]["mint"] == pytest.approx(
            math.log1p(1))

    def test_double_close_with_weights_bit_identical(self):
        batches = _scenario([
            _receipt_event(1, "agent-a"),
            _receipt_event(2, "agent-b"),
        ])
        kwargs = dict(
            agent_owner_map={"agent-a": "0xh1", "agent-b": "0xh2"},
            voice_weights={"0xh1": 0.123456789, "0xh2": 0.05},
        )
        r1 = federated_epoch_close(canonical_order(list(batches)), **kwargs)
        r2 = federated_epoch_close(canonical_order(list(batches)), **kwargs)
        for key in ("agent_mint", "tool_mint", "tool_positions",
                    "authoritative_payload"):
            assert json.dumps(r1[key]) == json.dumps(r2[key]), key


class TestVoiceOnDrift:
    def test_same_owner_reviews_collapse(self):
        """Two co-owned reviewers scoring the same axis pool into ONE
        household cell: mass log1p(2), value = their mean."""
        batches = _scenario([
            _receipt_event(1, "agent-a", axes={"correctness": 1.0}),
            _receipt_event(2, "agent-b", axes={"correctness": 0.0}),
        ])
        owner_map = {"agent-a": "0xhouse", "agent-b": "0xhouse"}
        result = federated_epoch_close(
            canonical_order(batches), agent_owner_map=owner_map)
        pos = result["tool_positions"][DIGEST]
        idx = 4  # correctness
        w = math.log1p(2)
        expected = (w * 0.5) / (1.0 + w)   # prior: zero head, mass 1.0
        assert pos["head"][idx] == pytest.approx(expected, rel=1e-6)
        assert pos["mass"][idx] == pytest.approx(1.0 + w, rel=1e-6)

    def test_weights_scale_drift_mass(self):
        """A heavier household drags the head further toward its score
        than a light one with the identical review."""
        def _close(weight: float):
            batches = _scenario([
                _receipt_event(1, "agent-a", axes={"correctness": 1.0}),
            ])
            return federated_epoch_close(
                canonical_order(batches),
                agent_owner_map={"agent-a": "0xh"},
                voice_weights={"0xh": weight},
            )["tool_positions"][DIGEST]["head"][4]

        heavy = _close(1.0)
        light = _close(0.05)
        assert heavy > light > 0.0
        w_h, w_l = math.log1p(1) * 1.0, math.log1p(1) * 0.05
        assert heavy == pytest.approx(w_h / (1.0 + w_h), rel=1e-6)
        assert light == pytest.approx(w_l / (1.0 + w_l), rel=1e-6)
