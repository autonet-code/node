"""Tool mint plumbing at the federated epoch close (docs/tool_substrate.md).

The mint itself is paid from concluded ToolBond review rounds (rules in
tests/test_review_mint.py). This file covers the plumbing around it:
registration attribution and carry-over, bit-identical and shuffled
closes, the owner-wallet keying, composition shares, the driver's
registration cache, and fees-pool conservation.
"""

from __future__ import annotations

import json
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
from nodes.common.world_model_substrate.charter import DEFAULT_CHARTER

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



E18 = 10 ** 18
R1, R2 = "0x" + "a1" * 20, "0x" + "b2" * 20
AXES = list(DEFAULT_CHARTER.ids)
SHARES = {R1: 0.3, R2: 0.2}


def _snap(digest: str = DIGEST, *, reviewers=(R1, R2)) -> Dict[str, Any]:
    """One concluding round for ``digest`` at rating 1.0."""
    revs = [{"reviewer": r, "stake": 0, "accuracy": E18,
             "axes": ["correctness"], "harm": False} for r in reviewers]
    return {
        "supply": 1000 * E18, "reviewer_share_bps": 4000,
        "tools": {digest: {"axes": {a: {"sum": 100 * E18, "mass": E18} for a in AXES},
                           "harm": {"sum": 0, "mass": 0}}},
        "rounds": {digest: [{"epoch": 1, "supply": 1000 * E18,
                             "prior_mass": {}, "reviews": revs}]},
    }


def _standard_batches() -> List[EventBatch]:
    return _batches([_registration_event(1)], Keypair.generate())


def _close(batches, **kw):
    kw.setdefault("embedding_dim", EMBED_DIM)
    kw.setdefault("emission_pool", 10.0)
    kw.setdefault("rep_shares", SHARES)
    kw.setdefault("review_snapshot", _snap())
    return federated_epoch_close(canonical_order(batches), **kw)


class TestToolMintClose:
    def test_close_mints_through_review_round(self):
        result = _close(_standard_batches())
        assert result["tool_registrations"][DIGEST]["author"] == AUTHOR
        assert result["tool_mint"][DIGEST]["author"] == AUTHOR
        assert result["agent_mint"].get(AUTHOR, 0) > 0
        assert result["agent_mint"].get(R1, 0) > 0

    def test_double_close_bit_identical(self):
        batches = _standard_batches()
        r1, r2 = _close(list(batches)), _close(list(batches))
        for key in ("agent_mint", "tool_mint", "tool_registrations",
                    "tool_rounds", "authoritative_payload"):
            assert json.dumps(r1[key]) == json.dumps(r2[key]), key

    def test_shuffled_delivery_deterministic(self):
        kp1, kp2 = Keypair.generate(), Keypair.generate()
        all_batches = (_batches([_registration_event(1)], kp1)
                       + _batches([_registration_event(1, digest="cd" * 32,
                                                       author="other")], kp2))
        rng = random.Random(11)
        results = []
        for _ in range(3):
            delivery = list(all_batches)
            rng.shuffle(delivery)
            results.append(_close(delivery))
        assert results[0]["tool_mint"] == results[1]["tool_mint"] == results[2]["tool_mint"]
        assert results[0]["agent_mint"] == results[1]["agent_mint"] == results[2]["agent_mint"]

    def test_attested_registration_carries_and_mints_nothing(self):
        batches = _batches([_registration_event(1, trust_class="attested")],
                           Keypair.generate())
        result = _close(batches)
        assert result["tool_mint"] == {}
        assert result["tool_registrations"][DIGEST]["trust_class"] == "attested"

    def test_mint_is_keyed_by_owner_wallet(self):
        """Decision 2026-10-27: the owner wallet claims. The anchored
        agent_mint names the author's lowercase owner, not the agent."""
        owner_map = {AUTHOR: "0xOwnerA", R1: "0xOwnerB", R2: "0xOwnerC"}
        # REP is read by household (owner wallet), so the shares are keyed by owner.
        result = _close(_standard_batches(), agent_owner_map=owner_map,
                        rep_shares={"0xownerb": 0.3, "0xownerc": 0.2})
        assert "0xownera" in result["agent_mint"]
        assert AUTHOR not in result["agent_mint"]
        assert result["agent_mint_by_agent"][AUTHOR] == result["agent_mint"]["0xownera"]
        assert result["authoritative_payload"]["agent_mint"] == result["agent_mint"]

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
    def _world(self):
        world = build_charter_world(embedding_dim=EMBED_DIM)
        apply_events(world, [_registration_event(1)], equilibrate_after=False)
        return world

    REGS = {DIGEST: {"trust_class": "pinned", "author": AUTHOR}}

    def test_carried_registration_attributes(self):
        """A round concluding in a later epoch attributes through the carried map."""
        out = compute_tool_mint(self._world(), [], registrations=self.REGS,
                                rep_shares=SHARES, review_snapshot=_snap())
        assert out["per_digest"][DIGEST]["author"] == AUTHOR
        assert out["per_digest"][DIGEST]["mint"] > 0
        assert out["registrations_next"][DIGEST]["author"] == AUTHOR

    def test_carried_registration_wins_over_reregistration(self):
        rereg = _registration_event(1)
        rereg["manifest_meta"] = {"trust_class": "pinned", "author": "mallory"}
        out = compute_tool_mint(self._world(), [rereg], registrations=self.REGS,
                                rep_shares=SHARES, review_snapshot=_snap())
        assert out["per_digest"][DIGEST]["author"] == AUTHOR

    def test_review_without_any_registration_mints_nothing(self):
        world = build_charter_world(embedding_dim=EMBED_DIM)
        out = compute_tool_mint(world, [], rep_shares=SHARES, review_snapshot=_snap())
        assert out["per_digest"] == {}
        assert out["registrations_next"] == {}


class TestCompositionFanOut:
    """Composition shares over the declared-dep DAG
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



def _plain_event() -> Dict[str, Any]:
    return {"kind": "observation_added", "seq": 1, "author_agent": "alice",
            "obs_id": "obs_x", "coords": _coords(), "label": "plain work"}


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
            [_registration_event(1)],                             # epoch 1
            [_plain_event()],                                     # epoch 2
        ])
        r1 = driver.run({"epoch_id": "e1"})
        assert r1.close_result["tool_registrations"][DIGEST]["author"] == AUTHOR

        # Epoch 2 has no events: the registration still carries.
        r2 = driver.run({"epoch_id": "e2"})
        assert r2.close_result["tool_registrations"][DIGEST]["author"] == AUTHOR

        # Restart: a new driver instance reloads the cache from disk.
        driver2 = self._driver(tmp_path, [[]])
        assert driver2._tool_registrations[DIGEST]["author"] == AUTHOR


# ---------------------------------------------------------------------------
# v4.1 "gradient trust" (ratified 2026-07-09), amended by fees-only +
# REP-from-earnings (Decision 2026-07-10):
#   DELETED: R4 supply-pegged beta cap, R6 rep/ATN split (mint is money only)
#   DELETED 2026-10-17 (docs/multi_jurisdiction.md decision 10): R5
#             reversal-aware credibility and gossip review drift; ratings
#             come from ToolBond. rep_shares weighs own_share only.
# ---------------------------------------------------------------------------



class TestFeesOnlyPool:
    """The emission pool is FEES-ONLY. A zero pool yields zero mint cleanly;
    a pool is conserved: minted plus carried-over equals the pool."""

    def test_zero_fee_epoch_zero_mint_clean(self):
        result = _close(_standard_batches(), emission_pool=0.0)
        assert result["agent_mint"] == {}
        assert result["total_mint"] == pytest.approx(0.0)

    def test_pool_conserved_exactly(self):
        pool = 12.5
        result = _close(_standard_batches(), emission_pool=pool)
        minted = sum(result["agent_mint"].values())
        carried = result["tool_rounds"]["pool_carry"] / 1e6
        assert minted > 0
        assert minted + carried == pytest.approx(pool, rel=1e-9)
        assert result["total_mint"] == pytest.approx(minted, rel=1e-9)
