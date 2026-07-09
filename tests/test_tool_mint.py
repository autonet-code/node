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

import pytest

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
                        digest: str = DIGEST, author: str = AUTHOR,
                        deps: str = "", axis: int = 4) -> Dict[str, Any]:
    meta = {"trust_class": trust_class, "author": author}
    if deps:
        meta["deps"] = deps
    return {
        "kind": "sub_claim_sprouted",
        "seq": seq,
        "author_agent": author,
        "tendency_id": "correctness",
        "parent_id": "solver_root",       # remaps to the live root
        "node_id": f"tm_{digest[:12]}",
        "position": "pro",
        "coords": _coords(axis=axis),
        "polarity_axis": _coords(axis=axis),
        "content": f"tool {digest[:8]}: does a thing",
        "author_post": True,              # immediate unit standing
        "artifact_digest": digest,
        "manifest_meta": meta,
    }


def _receipt_event(seq: int, caller: str, *, ok: bool = True,
                   digest: str = DIGEST,
                   attested: bool = True,
                   loadout: str = "") -> Dict[str, Any]:
    ev = {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": caller,
        "manifest_digest": digest,
        "tool_author": AUTHOR,
        "receipt_digest": f"r{seq:02d}" * 8,
        "ok": ok,
        "fee_atn": 0.0,
    }
    # Cognitive-attestation tier (the only mint input). Mechanical
    # receipts omit the fields entirely, like ToolUsed.to_dict does.
    if attested:
        ev["attested"] = True
        ev["score"] = 0.8
    if loadout:
        ev["loadout"] = loadout
    return ev


def _vet_event(seq: int, vetter: str, *, digest: str = DIGEST,
               ok: bool = True) -> Dict[str, Any]:
    """Third receipt flavor (spec: Vetting section): validator read the
    pinned code and attests code-adheres + no-malice. Not usage."""
    return {
        "kind": "tool_used",
        "seq": seq,
        "author_agent": vetter,
        "manifest_digest": digest,
        "tool_author": AUTHOR,
        "receipt_digest": f"v{seq:02d}" * 8,
        "ok": ok,
        "fee_atn": 0.0,
        "vet": True,
    }


def _vet_batches(digests: List[str], *, vetters: int = 2,
                 seq0: int = 1) -> List[EventBatch]:
    """Affirmative vets for each digest from ``vetters`` DISTINCT wire
    identities (each vetter = own keypair, so neither the wire dedup
    nor the fleet collapse bites). Two default vetters reach
    VET_QUORUM and greenlight in the same close."""
    out: List[EventBatch] = []
    for v in range(1, vetters + 1):
        events = [
            _vet_event(seq0 + i, f"vetter-{v}", digest=d)
            for i, d in enumerate(digests)
        ]
        out.extend(_batches(events, Keypair.generate()))
    return out


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


def _standard_batches(*, vetted: bool = True) -> List[EventBatch]:
    """Registration and receipts arrive from DIFFERENT signing keys —
    the wire-level self-attestation dedup excludes same-key callers, so
    a realistic mint scenario needs a distinct caller wire identity.
    ``vetted`` adds the distinct-fleet vets that greenlight the
    candidate (mint-eligibility, spec: Vetting section)."""
    kp_author = Keypair.generate()
    kp_callers = Keypair.generate()
    out = _batches([_registration_event(1)], kp_author) + _batches([
        _receipt_event(1, "caller-1"),
        _receipt_event(2, "caller-2"),
        _receipt_event(3, "caller-1", ok=False),   # failure: no mint credit
    ], kp_callers)
    if vetted:
        out += _vet_batches([DIGEST])
    return out


class TestToolMintClose:
    def test_close_mints_to_author(self):
        result = federated_epoch_close(canonical_order(_standard_batches()))
        tm = result["tool_mint"]
        assert DIGEST in tm
        entry = tm[DIGEST]
        assert entry["author"] == AUTHOR
        assert entry["ok_count"] == 2                       # failure excluded
        assert entry["mint"] > 0
        # Combo damper: two attesting agents with one ok attestation
        # each — usage_term = 2 * log1p(1), NOT log1p(total calls).
        assert entry["attesters"] == 2
        # v3: usage ALONE mints (no standing multiplier).
        assert entry["mint"] == pytest.approx(2 * math.log1p(1))
        assert "standing" not in entry
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
        all_batches = chain1 + chain2 + _vet_batches([DIGEST])
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

    def test_mechanical_receipts_do_not_mint(self):
        """Only cognitive attestations count — per-call exhaust is
        worth nothing (spec: Attestation section)."""
        kp_author, kp_caller = Keypair.generate(), Keypair.generate()
        batches = _batches([_registration_event(1)], kp_author) + _batches([
            _receipt_event(1, "caller-1", attested=False),
            _receipt_event(2, "caller-2", attested=False),
        ], kp_caller)
        result = federated_epoch_close(canonical_order(batches))
        assert result["tool_mint"] == {}

    def test_co_hosted_attestations_do_not_mint(self):
        """Wire-level dedup: attestation batches signed with the same
        key as the registration batch collapse to nothing — sybil
        agents on the author's own daemon can't pump usage."""
        kp = Keypair.generate()
        batches = _batches([
            _registration_event(1),
            _receipt_event(2, "sybil-1"),
            _receipt_event(3, "sybil-2"),
            _receipt_event(4, "sybil-3"),
        ], kp)  # ONE signing key for everything
        result = federated_epoch_close(canonical_order(batches))
        assert result["tool_mint"] == {}

    def test_same_owner_attestations_do_not_mint(self):
        """Owner-rooted exclusion (final form): callers registered under
        the same owner wallet as the author are the author's own fleet —
        their attestations don't count, even from a different daemon."""
        batches = _standard_batches()  # caller-1/caller-2 on separate key
        owner_map = {
            AUTHOR: "0xOwnerA",
            "caller-1": "0xOwnerA",     # same fleet as the author
            "caller-2": "0xOwnerB",     # genuine third party
        }
        result = federated_epoch_close(
            canonical_order(batches), agent_owner_map=owner_map)
        entry = result["tool_mint"][DIGEST]
        assert entry["attesters"] == 1          # only caller-2 counts
        assert entry["mint"] == pytest.approx(math.log1p(1))  # v3 usage-only

    def test_unknown_owner_is_not_excluded(self):
        """Absent map entries = unknown owner = no exclusion (the map is
        chain-derived; unregistered agents simply aren't in it)."""
        batches = _standard_batches()
        result = federated_epoch_close(
            canonical_order(batches),
            agent_owner_map={AUTHOR: "0xOwnerA"},  # callers unknown
        )
        assert result["tool_mint"][DIGEST]["attesters"] == 2

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
        world. v4.1: no greenlight needed to mint."""
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


def _vet_event_axes(seq: int, vetter: str, axes: Dict[str, float],
                    *, digest: str = DIGEST) -> Dict[str, Any]:
    """An INSPECTION REVIEW (v4.1 [R2]): a vet event carrying per-axis
    scores. Drifts position, mints nothing."""
    ev = _vet_event(seq, vetter, digest=digest)
    ev["axes"] = dict(axes)
    return ev


class TestVetGateRemoved:
    """v4.1 [R1][R2]: the vet GATE is gone — tools mint from first
    attested use, author keeps 100%, and a vet survives only as an
    inspection review that drifts position without minting."""

    REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}

    def _world(self):
        world = build_charter_world(embedding_dim=EMBED_DIM)
        apply_events(world, [_registration_event(1)], equilibrate_after=False)
        return world

    def test_unvetted_tool_mints(self):
        """v4.1 [R1]: no greenlight required — an attested, never-vetted
        tool mints from first use, author keeps everything."""
        result = federated_epoch_close(
            canonical_order(_standard_batches(vetted=False)))
        entry = result["tool_mint"][DIGEST]
        assert entry["author"] == AUTHOR
        assert entry["mint"] == pytest.approx(2 * math.log1p(1))
        # Author keeps 100% — no validator royalty split.
        node = result["agent_mint"]
        assert node.get(AUTHOR, 0) > 0

    def test_author_keeps_full_mint_no_royalty(self):
        """Even with vets present (now inspection reviews), the author's
        mint is not split with validators."""
        world = self._world()
        out = compute_tool_mint(
            world, [_receipt_event(1, "caller-9")],
            registrations=self.REGS,
        )
        entry = out["per_digest"][DIGEST]
        node = out["node_agent"][entry["node_id"]]
        assert node == {AUTHOR: pytest.approx(entry["mint"])}
        assert list(node.keys()) == [AUTHOR]

    def test_vet_inspection_review_drifts_but_mints_nothing(self):
        """v4.1 [R2]: a vet carrying axes moves the tool's charter head
        (position drift) but contributes ZERO usage/mint."""
        world = self._world()
        # A vet with a correctness score, from a rep-holding reviewer.
        vet = _vet_event_axes(1, "reviewer-1", {"correctness": 1.0})
        vet["sender"] = "aa01"
        out = compute_tool_mint(
            world, [vet],
            registrations=self.REGS,
            rep_shares={"reviewer-1": 0.5}, rep_supply=100.0,
        )
        # No usage -> no mint at all.
        assert out["per_digest"] == {}
        # ...but position drifted toward the inspection score.
        head = out["positions_next"][DIGEST]["head"]
        assert head[4] > 0.0     # correctness axis moved positive
        # And the review is recorded in the carried book for later
        # credibility re-scoring.
        assert DIGEST in out["tool_review_book_next"]
        assert "reviewer-1" in out["tool_review_book_next"][DIGEST]

    def test_vet_state_carried_tolerantly(self):
        """v4.1 [R1]: old carried vetting state (greenlit/royalty fields)
        is read without error and produces no mint effect."""
        world = self._world()
        legacy_vetting = {"manifests": {DIGEST: {
            "vets": {"v1": []}, "greenlit": True, "royalty_left": 5,
            "validators": ["v1"], "busted": False,
        }}, "busts": {}}
        out = compute_tool_mint(
            world, [_receipt_event(1, "caller-9")],
            registrations=self.REGS, vetting=legacy_vetting)
        entry = out["per_digest"][DIGEST]
        # Author still keeps 100% despite the legacy royalty window.
        node = out["node_agent"][entry["node_id"]]
        assert node == {AUTHOR: pytest.approx(entry["mint"])}


class TestCompositionFanOut:
    """Conservation of attestation over the declared-dep DAG
    (docs/tool_substrate.md — Composition rule 3)."""

    def test_shares_split_and_recursion(self):
        from nodes.common.federated_reconcile import _composition_shares
        deps_of = {"root": ["d1", "d2"], "d1": ["d3"], "d2": [], "d3": []}
        shares = _composition_shares("root", deps_of)
        assert shares["root"] == pytest.approx(0.7)
        assert shares["d2"] == pytest.approx(0.15)
        assert shares["d1"] == pytest.approx(0.15 * 0.7)
        assert shares["d3"] == pytest.approx(0.15 * 0.3)
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_cycle_and_missing_dep_forfeit(self):
        from nodes.common.federated_reconcile import _composition_shares
        # Cycle: d1 declares root back — that share is forfeited.
        shares = _composition_shares("root", {"root": ["d1"], "d1": ["root"]})
        assert shares["root"] == pytest.approx(0.7)
        assert shares["d1"] == pytest.approx(0.3 * 0.7)
        assert sum(shares.values()) < 1.0
        # Missing dep (declared, never registered): forfeits, never
        # redistributes to siblings.
        shares = _composition_shares("root", {"root": ["dx", "d2"], "d2": []})
        assert shares["root"] == pytest.approx(0.7)
        assert shares["d2"] == pytest.approx(0.15)
        assert "dx" not in shares

    def test_close_fans_attestation_over_deps(self):
        """One attestation of a composite credits the composite AND its
        declared dep — royalty flow, conserved."""
        dep_digest = "cc" * 32
        kp_authors, kp_caller = Keypair.generate(), Keypair.generate()
        batches = _batches([
            _registration_event(1, digest=dep_digest, author="dep-author",
                                axis=2),
            _registration_event(2, digest=DIGEST, author=AUTHOR,
                                deps=dep_digest, axis=4),
        ], kp_authors) + _batches([
            _receipt_event(1, "caller-1"),        # attests the COMPOSITE
        ], kp_caller) + _vet_batches([DIGEST, dep_digest])
        result = federated_epoch_close(canonical_order(batches))
        tm = result["tool_mint"]
        assert DIGEST in tm and dep_digest in tm
        # Damp-then-split: log1p applies ONCE (at the attested
        # composite), the damped value distributes linearly — the
        # anti-amplification order of operations.
        damped = math.log1p(1)
        assert tm[DIGEST]["usage_term"] == pytest.approx(damped * 0.7)
        assert tm[dep_digest]["usage_term"] == pytest.approx(damped * 0.3)
        assert tm[dep_digest]["author"] == "dep-author"
        # Conservation of the damped quantity across the DAG.
        assert (tm[DIGEST]["usage_term"] + tm[dep_digest]["usage_term"]
                == pytest.approx(damped))
        # v3: mint = usage share directly (no standing multiplier).
        assert tm[DIGEST]["mint"] == pytest.approx(damped * 0.7)
        assert tm[dep_digest]["mint"] == pytest.approx(damped * 0.3)

    def test_self_dep_padding_cannot_amplify(self):
        """An author padding a composite with their own junk deps moves
        credit between their tools but total author mint never exceeds
        the un-padded case (same standing assumed)."""
        junk = "ee" * 32
        kp_authors, kp_caller = Keypair.generate(), Keypair.generate()
        padded = _batches([
            _registration_event(1, digest=junk, author=AUTHOR, axis=2),
            _registration_event(2, digest=DIGEST, author=AUTHOR,
                                deps=junk, axis=4),
        ], kp_authors) + _batches(
            [_receipt_event(1, "caller-1")], kp_caller,
        ) + _vet_batches([DIGEST, junk])
        plain = _batches([
            _registration_event(1, digest=DIGEST, author=AUTHOR, axis=4),
        ], Keypair.generate()) + _batches(
            [_receipt_event(1, "caller-1")], Keypair.generate(),
        ) + _vet_batches([DIGEST])

        r_padded = federated_epoch_close(canonical_order(padded))
        r_plain = federated_epoch_close(canonical_order(plain))
        # log1p is concave: log1p(0.7)+log1p(0.3) < 2*log1p(0.5) but
        # more importantly <= ... the padded author total must not
        # exceed the plain total given equal standings (both nodes
        # standing ~1 from the author post).
        padded_total = sum(e["mint"] for e in r_padded["tool_mint"].values())
        plain_total = sum(e["mint"] for e in r_plain["tool_mint"].values())
        assert padded_total <= plain_total + 1e-9


class TestLoadoutAdoption:
    """Resident grammar: distros earn by distinct-fleet adoption,
    volume-blind, fanned over their module deps."""

    MODULE = "11" * 32
    DISTRO = "22" * 32

    def _batches_with_adoption(self):
        kp_auth = Keypair.generate()
        kp_f1, kp_f2 = Keypair.generate(), Keypair.generate()
        reg = _batches([
            _registration_event(1, digest=self.MODULE, author="mod-author",
                                axis=2),
            _registration_event(2, digest=self.DISTRO, author="distro-author",
                                deps=self.MODULE, axis=3),
            _registration_event(3, digest=DIGEST, author=AUTHOR, axis=4),
        ], kp_auth)
        # Fleet 1: FIVE attestations under the distro (volume must not
        # matter for adoption). Fleet 2: one.
        f1 = _batches([
            _receipt_event(i, "caller-1", loadout=self.DISTRO)
            for i in range(1, 6)
        ], kp_f1)
        f2 = _batches([
            _receipt_event(1, "caller-2", loadout=self.DISTRO),
        ], kp_f2)
        return reg + f1 + f2 + _vet_batches([self.MODULE, self.DISTRO, DIGEST])

    def test_adoption_volume_blind_and_fanned(self):
        owner_map = {
            AUTHOR: "0xA", "mod-author": "0xM", "distro-author": "0xD",
            "caller-1": "0xF1", "caller-2": "0xF2",
        }
        result = federated_epoch_close(
            canonical_order(self._batches_with_adoption()),
            agent_owner_map=owner_map)
        tm = result["tool_mint"]
        ln2 = math.log1p(1)
        # Two distinct fleets, log1p(1) each, split 0.7 distro / 0.3 module.
        assert tm[self.DISTRO]["usage_term"] == pytest.approx(2 * ln2 * 0.7)
        assert tm[self.MODULE]["usage_term"] == pytest.approx(2 * ln2 * 0.3)
        # Invoked credit for the tool itself stays volume-SENSITIVE:
        # caller-1 attested 5 times, caller-2 once.
        assert tm[DIGEST]["usage_term"] == pytest.approx(
            math.log1p(5) + math.log1p(1))

    def test_same_fleet_as_distro_author_excluded_but_module_credited(self):
        """A fleet can't pump its own distro — but its adoption still
        credits upstream modules it doesn't own (exclusions are
        per-target)."""
        owner_map = {
            AUTHOR: "0xA", "mod-author": "0xM", "distro-author": "0xD",
            "caller-1": "0xD",   # same fleet as the distro author
            "caller-2": "0xF2",
        }
        result = federated_epoch_close(
            canonical_order(self._batches_with_adoption()),
            agent_owner_map=owner_map)
        tm = result["tool_mint"]
        ln2 = math.log1p(1)
        # Distro: only fleet 2 counts.
        assert tm[self.DISTRO]["usage_term"] == pytest.approx(1 * ln2 * 0.7)
        # Module: both fleets count (module author's owner differs).
        assert tm[self.MODULE]["usage_term"] == pytest.approx(2 * ln2 * 0.3)


class TestDriverCarryOver:
    """FederatedCloseDriver accumulates + persists tool registrations
    across epochs (and across a driver restart)."""

    def _driver(self, tmp_path, events_per_epoch):
        from unittest.mock import MagicMock

        from nodes.common.event_gossip import EventGossip
        from nodes.common.federated_close_driver import FederatedCloseDriver

        kp = Keypair.generate()
        gossip = MagicMock(spec=EventGossip)
        gossip.known_senders.return_value = [kp.public_key]
        gossip.sender_pubkey = kp.public_key
        gossip.drain_epoch_batches.side_effect = [
            _batches(evs, kp) for evs in events_per_epoch
        ]
        return FederatedCloseDriver(
            gossip=gossip, embedding_dim=EMBED_DIM,
            tool_registrations_path=tmp_path / "tool_registrations.json",
        )

    def test_registration_carries_across_epochs_and_restart(self, tmp_path):
        driver = self._driver(tmp_path, [
            [_registration_event(1), _receipt_event(2, "c1")],   # epoch 1
            [_receipt_event(1, "c2")],                            # epoch 2
        ])
        r1 = driver.run({"epoch_id": "e1"})
        assert r1.close_result["tool_registrations"][DIGEST]["author"] == AUTHOR

        # Epoch 2 has receipts only — the carried registration still
        # attributes (per_digest empty here only because the fresh
        # charter world lacks the claim node; the REGISTRATION survives).
        r2 = driver.run({"epoch_id": "e2"})
        assert r2.close_result["tool_registrations"][DIGEST]["author"] == AUTHOR

        # Restart: a new driver instance reloads the cache from disk.
        driver2 = self._driver(tmp_path, [[_receipt_event(1, "c3")]])
        assert driver2._tool_registrations[DIGEST]["author"] == AUTHOR


# ---------------------------------------------------------------------------
# v4.1 "gradient trust" (ratified 2026-07-09):
#   R4 supply-pegged beta cap on zero-rep mint weight
#   R6 rep/ATN split (agent_rep decoupled from agent_mint)
#   R5 continuous reversal-aware credibility (drift-weight multiplier)
# ---------------------------------------------------------------------------


def _receipt_axes(seq: int, caller: str, axes: Dict[str, float],
                  *, digest: str = DIGEST) -> Dict[str, Any]:
    ev = _receipt_event(seq, caller, digest=digest)
    ev["axes"] = dict(axes)
    return ev


class TestBetaCap:
    """R4: zero-rep households' AGGREGATE mint weight is capped at
    beta(rep_supply) of total weight; rep_supply=None => uncapped."""

    REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}

    def _world(self):
        world = build_charter_world(embedding_dim=EMBED_DIM)
        apply_events(world, [_registration_event(1)], equilibrate_after=False)
        return world

    def test_uncapped_when_rep_supply_none(self):
        """rep_shares=None => local regime, beta=1, no scaling."""
        world = self._world()
        out = compute_tool_mint(
            world, [_receipt_event(1, "zero-1"), _receipt_event(2, "zero-2")],
            registrations=self.REGS,
        )
        assert out["beta"] == pytest.approx(1.0)
        assert out["per_digest"][DIGEST]["mint"] == pytest.approx(
            2 * math.log1p(1))

    def test_zero_rep_aggregate_scaled_when_cap_binds(self):
        """With a tiny rep base and many zero-rep callers, the zero-rep
        aggregate mint weight is scaled down to beta of total."""
        world = self._world()
        from nodes.common.federated_reconcile import (
            BETA_MIN, VOICE_EPSILON, beta_of_supply)
        rep_supply = 100000.0          # mature: beta floored at BETA_MIN
        beta = beta_of_supply(rep_supply)
        assert beta == pytest.approx(BETA_MIN)
        events = [
            _receipt_event(1, "rep-holder"),
            _receipt_event(2, "zero-1"),
            _receipt_event(3, "zero-2"),
            _receipt_event(4, "zero-3"),
        ]
        rep_shares = {"rep-holder": 0.5}
        out = compute_tool_mint(
            world, events, registrations=self.REGS,
            rep_shares=rep_shares, rep_supply=rep_supply,
        )
        assert out["beta"] == pytest.approx(beta)
        ln = math.log1p(1)
        rep_w = ln * 0.5
        zero_w_raw = 3 * ln * VOICE_EPSILON
        allowed = (beta / (1.0 - beta)) * rep_w
        assert allowed < zero_w_raw, "cap should bind in this fixture"
        expected = rep_w + allowed
        assert out["per_digest"][DIGEST]["mint"] == pytest.approx(
            expected, rel=1e-6)

    def test_zero_rep_below_cap_unchanged(self):
        """When zero-rep aggregate is under beta, no scaling applies."""
        world = self._world()
        from nodes.common.federated_reconcile import VOICE_EPSILON
        rep_supply = 100.0
        events = [
            _receipt_event(1, "rep-holder"),
            _receipt_event(2, "zero-1"),
        ]
        rep_shares = {"rep-holder": 0.9}
        out = compute_tool_mint(
            world, events, registrations=self.REGS,
            rep_shares=rep_shares, rep_supply=rep_supply,
        )
        ln = math.log1p(1)
        expected = ln * 0.9 + ln * VOICE_EPSILON
        assert out["per_digest"][DIGEST]["mint"] == pytest.approx(
            expected, rel=1e-6)


class TestRepSplit:
    """R6: agent_rep = agent_mint x rep_fraction; zero-rep-only callers
    grant ATN but ZERO reputation."""

    REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}

    def _world(self):
        world = build_charter_world(embedding_dim=EMBED_DIM)
        apply_events(world, [_registration_event(1)], equilibrate_after=False)
        return world

    def test_rep_fraction_mixed_callers(self):
        world = self._world()
        from nodes.common.federated_reconcile import VOICE_EPSILON
        events = [
            _receipt_event(1, "rep-holder"),
            _receipt_event(2, "zero-1"),
        ]
        rep_shares = {"rep-holder": 0.5}
        out = compute_tool_mint(
            world, events, registrations=self.REGS,
            rep_shares=rep_shares, rep_supply=1.0,   # beta ~= 1, no cap
        )
        entry = out["per_digest"][DIGEST]
        ln = math.log1p(1)
        rep_part = ln * 0.5
        total = rep_part + ln * VOICE_EPSILON
        assert entry["rep_fraction"] == pytest.approx(
            rep_part / total, rel=1e-6)
        node = entry["node_id"]
        assert out["node_agent_rep"][node][AUTHOR] == pytest.approx(rep_part)

    def test_zero_rep_only_grants_no_reputation(self):
        world = self._world()
        events = [_receipt_event(1, "zero-1"), _receipt_event(2, "zero-2")]
        # Low rep_supply => beta ~= 1 (uncapped, genesis regime): zero-rep
        # usage STILL mints ATN, but rep_shares={} means the rep basis is
        # zero — ATN minted, reputation not.
        out = compute_tool_mint(
            world, events, registrations=self.REGS,
            rep_shares={}, rep_supply=1.0,
        )
        entry = out["per_digest"][DIGEST]
        assert entry["mint"] > 0
        assert entry["rep_fraction"] == pytest.approx(0.0)
        assert out["node_agent_rep"] == {}

    def test_agent_rep_le_agent_mint_through_close(self):
        """Through the full close, agent_rep <= agent_mint per agent."""
        kp_author = Keypair.generate()
        kp_caller = Keypair.generate()
        batches = _batches([_registration_event(1)], kp_author) + _batches([
            _receipt_event(1, "rep-caller"),
            _receipt_event(2, "zero-caller"),
        ], kp_caller)
        result = federated_epoch_close(
            canonical_order(batches),
            rep_shares={"rep-caller": 0.5},
            rep_supply=1.0,            # beta ~= 1, isolate the split
        )
        am = result["agent_mint"]
        ar = result.get("agent_rep", {})
        assert am.get(AUTHOR, 0) > 0
        for agent, rep in ar.items():
            assert rep <= am[agent] + 1e-9
        assert ar.get(AUTHOR, 0) > 0

    def test_zero_rep_only_close_grants_no_agent_rep(self):
        kp_author = Keypair.generate()
        kp_caller = Keypair.generate()
        batches = _batches([_registration_event(1)], kp_author) + _batches([
            _receipt_event(1, "zero-1"),
        ], kp_caller)
        result = federated_epoch_close(
            canonical_order(batches),
            rep_shares={}, rep_supply=1.0,
        )
        assert result["agent_mint"].get(AUTHOR, 0) > 0
        assert result.get("agent_rep", {}).get(AUTHOR, 0) == 0


class TestCredibility:
    """R5: continuous reversal-aware credibility carried across closes;
    a reversal retroactively docks the capturers."""

    REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}

    def _world(self):
        world = build_charter_world(embedding_dim=EMBED_DIM)
        apply_events(world, [_registration_event(1)], equilibrate_after=False)
        return world

    def _many(self, caller: str, axis_val: float, n: int, seq0: int):
        """n attested receipts from one caller, each scoring correctness
        — high n lifts log1p(n) so review mass clears CRED_MASS_FLOOR and
        the drift centroid is well-defined."""
        return [_receipt_axes(seq0 + i, caller, {"correctness": axis_val})
                for i in range(n)]

    def test_credibility_carries_and_reversal_docks(self):
        world = self._world()
        # Heavy honest majority + a lone capturer. rep_supply=1 => beta~=1
        # (no cap distortion), rep shares near 1 so drift mass is large.
        rep_shares = {"honest-1": 0.9, "honest-2": 0.9, "capturer": 0.9}
        # Epoch 1: EVERYONE (incl. capturer) asserts strongly POSITIVE,
        # many reviews each so total mass >> CRED_MASS_FLOOR.
        ev1 = (self._many("honest-1", 1.0, 30, 1)
               + self._many("honest-2", 1.0, 30, 100)
               + self._many("capturer", 1.0, 30, 200))
        out1 = compute_tool_mint(
            world, ev1, registrations=self.REGS,
            rep_shares=rep_shares, rep_supply=1.0,
        )
        cred1 = out1["credibility_next"]
        book1 = out1["tool_review_book_next"]
        assert DIGEST in book1
        # Agreement so far: nobody docked.
        for h in ("honest-1", "honest-2", "capturer"):
            assert cred1.get(h, 1.0) >= 0.9

        # Epoch 2: the honest majority REVERSES hard to strongly negative
        # with overwhelming mass, dragging the head far below the
        # capturer's stored +1.0 review -> the capturer is retroactively
        # docked (its carried centroid now deviates > delta from the head).
        ev2 = (self._many("honest-1", -1.0, 200, 1)
               + self._many("honest-2", -1.0, 200, 300))
        out2 = compute_tool_mint(
            world, ev2,
            registrations=self.REGS,
            rep_shares=rep_shares, rep_supply=1.0,
            positions=out1["positions_next"],
            credibility=cred1,
            tool_review_book=book1,
        )
        head2 = out2["positions_next"][DIGEST]["head"][4]
        assert head2 < 0.0, "honest reversal should push head negative"
        cred2 = out2["credibility_next"]
        assert cred2.get("capturer", 1.0) < 1.0

    def test_credibility_floor_and_recovery(self):
        """A docked household recovers toward 1.0 in a later quiet epoch,
        never below CRED_FLOOR."""
        from nodes.common.federated_reconcile import CRED_FLOOR
        world = self._world()
        cred_in = {"stale": CRED_FLOOR}
        out = compute_tool_mint(
            world, [_receipt_event(1, "someone")],
            registrations=self.REGS,
            rep_shares={"someone": 0.5}, rep_supply=100000.0,
            credibility=cred_in,
        )
        c = out["credibility_next"].get("stale", CRED_FLOOR)
        assert CRED_FLOOR <= c <= 1.0
        assert c > CRED_FLOOR
