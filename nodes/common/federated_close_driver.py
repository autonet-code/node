"""Federated-close driver — runs at epoch boundaries to produce the
deterministic authoritative result and pick the on-chain submitter.

The pieces this glues together (all already exist and tested):

  - ``EventGossip.drain_epoch_batches()`` — gather all batches that
    arrived (own + peer) in the closing epoch.
  - ``canonical_order(batches)`` — sender-grouped Merkle-rooted
    deterministic ordering. Bit-identical across honest daemons that
    received the same batch set.
  - ``federated_epoch_close(canonical, ...)`` — replay the canonical
    sequence on a fresh charter world; produce a result whose
    ``authoritative_payload`` is bit-identical too.

The driver also picks **a single submitter** for the on-chain anchor.
Daemons run hash(epoch_id || canonical_root) modulo the canonical
sender set; the resulting pubkey is the winner. Honest daemons all
compute the same winner without coordination — no extra round-trip,
no leader election. Substrate.sol's ``isAnchored(epoch_id)`` rejects
duplicates anyway, so a tied or buggy selection is non-fatal.

A fallback timer lets the next-in-line submit if the winner doesn't
land an anchor within ``fallback_seconds``. This handles a winner
going offline between the canonical close and the anchor submission.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .authoritative_encoding import (
    CARRY_KEYS,
    cid_for_blob,
    encode_carry_bundle,
)
from .canonical_ordering import canonical_order
from .event_gossip import EventBatch, EventGossip
from .federated_reconcile import federated_epoch_close
from .world_model_substrate.adapter import build_charter_world


logger = logging.getLogger(__name__)


@dataclass
class FederatedCloseResult:
    """What the driver produces at one epoch close."""
    epoch_id: str
    close_result: Dict[str, Any]
    senders: List[bytes]
    winner: bytes
    is_winner: bool
    n_batches: int
    # State sync (task: anchored catch-up). When a canonical tracker
    # is wired, every daemon computes the same cumulative-canonical-
    # world checkpoint blob; its cid rides in the authoritative
    # payload's world_cid field, and the blob itself is published to
    # the blob resolver by the chain submission driver.
    world_cid: str = ""
    world_checkpoint_blob: bytes = b""
    # Verifiable carry-over (payload schema 4). Every daemon computes the
    # same canonical bundle of the three tool maps this close produced
    # (the next close's inputs); its cid rides the payload's carry_cid
    # field and every daemon publishes the blob, so a joiner can fetch
    # and verify it against the anchor instead of trusting a peer.
    carry_cid: str = ""
    carry_bundle_blob: bytes = b""
    # Anchor authority + conservation (2026-10-12): the chain anchorCount
    # the voice/pool snapshot was read at. The close's pool equals the
    # on-chain mintPool only if it is anchored as anchor number
    # snapshot_anchor_count; signers refuse to sign otherwise. None = no
    # chain source (local/test).
    snapshot_anchor_count: Optional[int] = None
    # Pipelined elections (Decision 2026-10-26): this epoch's number and the
    # root of the epoch before it as THIS daemon closed it (its parent),
    # which may not be anchored yet. None/"" outside the pipelined mode.
    epoch_number: Optional[int] = None
    parent_root: str = ""


def pick_submitter(
    epoch_id: str,
    senders: List[bytes],
    canonical_root: bytes = b"",
) -> Optional[bytes]:
    """Deterministic-random submitter selection.

    Pure function of ``(epoch_id, canonical_root, sender set)`` so all
    daemons compute the same answer. Returns one of ``senders``, or
    None if no senders.
    """
    if not senders:
        return None
    sorted_senders = sorted(senders)
    h = hashlib.sha256()
    h.update(b"autonet:submitter:v1")
    h.update(epoch_id.encode("utf-8"))
    h.update(canonical_root)
    digest = h.digest()
    idx = int.from_bytes(digest[:8], "big") % len(sorted_senders)
    return sorted_senders[idx]


class FederatedCloseDriver:
    """Per-daemon driver invoked from the WorldService epoch-close
    subscriber. Stateless across epochs — each call builds a fresh
    canonical order from the gossip's per-epoch buffer.
    """

    def __init__(
        self,
        gossip: EventGossip,
        *,
        bandwidth: float = 1.5,
        embedding_dim: int = 1024,
        canonical_tracker: Optional[Any] = None,
        pricing: str = "ledger",
        tool_registrations_path: Optional[Any] = None,
        tool_vetting_path: Optional[Any] = None,
        tool_positions_path: Optional[Any] = None,
        canonical_world_path: Optional[Any] = None,
    ):
        self.gossip = gossip
        self.pricing = pricing
        self.bandwidth = bandwidth
        self.embedding_dim = embedding_dim
        # Optional state_sync.CanonicalWorldTracker. When present, each
        # close also advances the cumulative canonical world and embeds
        # its checkpoint cid in the authoritative payload (world_cid).
        # Must be constructed with the SAME bandwidth/embedding_dim so
        # the tracker's replay matches the federated kernel.
        self.canonical_tracker = canonical_tracker
        # The tracker's latest checkpoint blob, persisted after every close
        # and restored at boot (Decision 2026-10-26): a tracker rebuilt from
        # a fresh charter after a restart emits a world_cid no peer shares.
        self._canonical_world_path: Optional[Path] = (
            Path(canonical_world_path) if canonical_world_path else None
        )
        # True once the tracker holds the network's canonical world: a
        # restored or chain-fetched checkpoint, or a fresh charter at
        # genesis. A failed tracker update clears it (its world is then
        # half-advanced).
        self._tracker_restored = self._restore_canonical_tracker()
        self._tracker_failed: Optional[str] = None
        # Optional zero-arg hook returning a state_sync.CatchUpResult for
        # the latest anchor's world_cid (None when not retrievable), used
        # while the tracker has no checkpoint past genesis. Same install
        # rule as the carry-over: only the epoch observed just before.
        self.world_source: Optional[Callable[[], Any]] = None
        # Set when the latest voice refresh failed: closing on the previous
        # snapshot would price this epoch on state peers have moved past.
        self._voice_refresh_error: Optional[str] = None
        # The epoch window after the latest anchor (Decision 2026-10-26),
        # from the voice refresh (voice_state.epoch_window). None = no
        # chain window: the library/local behavior (every buffered batch,
        # the local epoch id).
        self.epoch_window: Optional[Dict[str, Any]] = None
        # Optional zero-arg hook returning the chain's anchorCount. Every
        # voice-refresh input is pinned to the latest anchor, so while the
        # count is unchanged the refresh is skipped.
        self.anchor_count_source: Optional[Callable[[], int]] = None
        self.clock: Callable[[], float] = time.time
        # One epoch in flight: the number of the epoch this daemon last
        # closed. It closes no other until the anchor before the next one
        # lands. Persisted so a restart mid-flight does not close the
        # same epoch again with a different batch set.
        self._closed_epoch_number: Optional[int] = None
        # Pipelined mode (Decision 2026-10-26, items 5-6), on when this
        # zero-arg hook is wired. It returns {"anchor_count": N, "anchors":
        # [{"number", "epoch_id", "root", "cutoff"}, ...]} for the latest
        # anchors (oldest first). Epochs then close on their own schedule,
        # each built on this daemon's own view of the unanchored ones
        # before it, every input pinned to the previous epoch's cutoff.
        self.chain_source: Optional[Callable[[], Dict[str, Any]]] = None
        # (from_ts, to_ts) -> ATN burned by fees between them, from
        # Substrate's on-chain checkpoints: with a target share set, an
        # epoch ends on that volume (voice_state.volume_cutoff).
        self.burned_source: Optional[Callable[[int, int], int]] = None
        # Times this daemon found an epoch anchored differently from its own
        # close and rewound onto the anchored state (a run collector metric).
        self.rewinds: int = 0
        # This daemon's closed but unanchored epochs, oldest first:
        # {"number", "epoch_id", "cutoff", "root"} plus their member batches
        # (in memory only), so a rewind can put them back in the buffer.
        self._local_epochs: List[Dict[str, Any]] = []
        self._rewind_pending = False
        self._pin_key: Optional[tuple] = None
        # Tool-substrate carry-over (docs/tool_substrate.md v2): the
        # accumulated digest -> manifest_meta registration map, so a
        # tool registered in epoch 1 still attributes mint in epoch 5.
        # Derived purely from canonical events (each close returns the
        # advanced map), so the on-disk copy is a CACHE — rebuildable
        # by replaying epochs, identical on every honest daemon.
        self._tool_registrations_path: Optional[Path] = (
            Path(tool_registrations_path) if tool_registrations_path else None
        )
        self._tool_registrations: Dict[str, Dict[str, str]] = (
            self._load_tool_registrations()
        )
        # Vetting carry-over (spec: Vetting section): candidate/greenlit
        # state + validator bust counts. Same contract as the
        # registrations map — derived purely from canonical events and
        # replayed world state, so the on-disk copy is a rebuildable
        # cache, identical on every honest daemon.
        self._tool_vetting_path: Optional[Path] = (
            Path(tool_vetting_path) if tool_vetting_path else None
        )
        self._tool_vetting: Dict[str, Any] = self._load_tool_vetting()
        # v3 position-drift carry-over (spec Decision 2026-07-08):
        # digest -> {"head": [n_dims], "mass": [n_dims]}: the mint-weighted review
        # centroid state. Same contract as registrations/vetting: derived
        # purely from canonical events, the on-disk copy is a rebuildable
        # cache, identical on every honest daemon.
        self._tool_positions_path: Optional[Path] = (
            Path(tool_positions_path) if tool_positions_path else None
        )
        self._tool_positions: Dict[str, Dict[str, Any]] = (
            self._load_tool_positions()
        )
        # Owner-rooted damper exclusion (spec: Owner-rooted registration):
        # agent id -> owner wallet, sourced from chain owner-binding data
        # (OwnerBound events / getAgentOwner). Every daemon reading
        # the same anchored chain state derives the same map, so the
        # bit-identical close guarantee holds. Empty until the chain
        # sourcing is wired (registerAgent v2 deploy) — the wire-level
        # batch-key dedup remains the interim floor.
        self.agent_owner_map: Dict[str, str] = {}
        # Household voice weights (spec: balance-weighted voice addendum):
        # household key (owner wallet, or agent id when unbound) ->
        # epsilon + household_ATN / supply. Same trust contract as the
        # owner map: chain-derived, identical at every daemon reading the
        # same anchored state. Empty map -> close runs with weights=None
        # (every household weighs 1.0, the no-chain behavior).
        self.voice_weights: Dict[str, float] = {}
        # v4.1 gradient trust (amended 2026-07-10): the RAW reputation
        # share per household (rep/rep_supply, NO epsilon floor), read from
        # RepToken checkpoints (DAO-side). Distinct from voice_weights
        # (which carries the floor for mint bootstrap). Drift weight needs
        # the un-floored share — the β cap and rep/ATN split are gone, so
        # rep_supply is no longer threaded. Empty => local regime
        # (weight-1.0 drift). Comes from the same voice_state read.
        self.rep_shares: Dict[str, float] = {}
        # Fees-only emission pool (Decision 2026-07-10): the burned service
        # fees in the snapshot anchor's window (NO base floor), computed by
        # the same voice_source refresh. None (no chain source) = no
        # normalization — raw mint units, the legacy/local behavior; a zero
        # pool yields zero mint.
        self.emission_pool: Optional[float] = None
        # anchorCount at the last successful voice/pool refresh.
        self.snapshot_anchor_count: Optional[int] = None
        # ToolBond review snapshot (docs/multi_jurisdiction.md decision 10):
        # tool ratings and the reviewer-share rounds, rebuilt from ToolBond
        # logs pinned to the same snapshot anchor block, by the same
        # voice_source refresh. None (no chain source / no ToolBond) = zero
        # ratings and no reviewer shares.
        self.review_snapshot: Optional[Dict[str, Any]] = None
        # REP total supply at the same pin (containment tiers weigh review
        # mass against it, atn/tool_security.py).
        self.rep_supply: int = 0
        # The jurisdiction's charter text pinned to the snapshot anchor
        # (docs/multi_jurisdiction.md decision 5), read by the same
        # voice_source refresh. None = no pinned read (no DAO Registry
        # configured, genesis, or never written).
        self.pinned_charter_text: Optional[str] = None
        # Households under a DAO sanction at the snapshot anchor
        # (docs/accountability.md), by the same voice_source refresh. Their
        # mint is withheld and their estimates carry no drift weight.
        self.sanctioned_households: List[str] = []
        # Set to the new charter text once a pinned charter differs from
        # the active one: the close must not run on the old axes. The host
        # restarts the daemon onto the new charter (on_charter_change).
        self.charter_restart_required: Optional[str] = None
        # Optional one-arg callback(new_charter_text), called once when
        # ``charter_restart_required`` is first set (from the close thread).
        self.on_charter_change: Optional[Callable[[str], None]] = None
        # Optional zero-arg refresh hook returning
        # {"owner_map": {...}, "voice_weights": {...}} — wired by the
        # host when chain access exists (see AutonetService.
        # attach_chain_submission). Called at the top of each run so the
        # close prices this epoch's voices from current chain state; on
        # failure the previous maps stand (stale beats forked).
        self.voice_source: Optional[Any] = None
        # Participation preconditions (off by default so library/test use
        # is unchanged; AutonetService turns it on). When on, run()
        # refuses to close (log + skip) while a close here would fork
        # from peers: no chain read access, or missing carry-over state.
        self.enforce_preconditions: bool = False
        # Set by the latest successful voice refresh: True when the chain
        # has no anchors yet (genesis, empty carry-over is correct),
        # False when it has anchors, None when unknown.
        self._chain_at_genesis: Optional[bool] = None
        # Joiner bootstrap (payload schema 4): zero-arg hook returning a
        # state_sync.CarryOverFetch for the latest anchor (wired by the
        # host from chain read access + the blob resolver). Consulted
        # only while carry-over files are missing past genesis; the
        # daemon observes (refuses to close) until it yields a verified
        # bundle.
        self.carry_source: Optional[Callable[[], Any]] = None
        # Canonical epoch_root (hex) of the most recent epoch whose
        # batches this daemon drained, closed or merely observed. A joiner
        # installs an anchored bundle only when the anchor's epoch_root
        # equals this value at the next close: that proves the bundle is
        # the output of the epoch IMMEDIATELY before the one being closed
        # (epoch ids are per-daemon, so roots are the shared identity).
        self._last_observed_root: Optional[str] = None
        # Set while the latest anchor may be a genesis seed this daemon has
        # not installed (bundle unretrievable or rejected): closing on the
        # pre-seed local state would fork from peers that adopted it.
        self._seed_blocker: Optional[str] = None
        self._closed_epoch_number = self._load_in_flight()
        self._local_epochs = self._load_local_epochs()

    def _in_flight_path(self) -> Optional[Path]:
        base = self._tool_registrations_path or self._canonical_world_path
        return None if base is None else base.parent / "epoch_in_flight.json"

    def _load_in_flight(self) -> Optional[int]:
        path = self._in_flight_path()
        if path is None or not path.exists():
            return None
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["number"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save_in_flight(self, number: int, epoch_id: str) -> None:
        path = self._in_flight_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps({"number": int(number),
                                       "epoch_id": str(epoch_id)}),
                           encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist the in-flight epoch: %s", e)

    # ---- pipelined elections: the local epoch chain -------------------

    def _local_chain_path(self) -> Optional[Path]:
        base = self._tool_registrations_path or self._canonical_world_path
        return None if base is None else base.parent / "epoch_chain.json"

    def _load_local_epochs(self) -> List[Dict[str, Any]]:
        path = self._local_chain_path()
        if path is None or not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [{"number": int(e["number"]), "epoch_id": str(e["epoch_id"]),
                     "cutoff": int(e["cutoff"]), "root": str(e["root"]),
                     "members": []} for e in data]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def _save_local_epochs(self) -> None:
        path = self._local_chain_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps([
                {k: e[k] for k in ("number", "epoch_id", "cutoff", "root")}
                for e in self._local_epochs]), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist the local epoch chain: %s", e)

    @property
    def local_epochs(self) -> List[Dict[str, Any]]:
        return [{k: e[k] for k in ("number", "epoch_id", "cutoff", "root")}
                for e in self._local_epochs]

    def _reconcile(self, anchor_count: int,
                   anchored: Dict[int, Dict[str, Any]]) -> None:
        """Drop local epochs the chain anchored as this daemon closed them;
        on the first one it anchored differently, rewind: every local epoch
        is void (each was built on the one before), its batches go back to
        the buffer, and the anchored state is installed before closing."""
        while self._local_epochs and self._local_epochs[0]["number"] <= anchor_count:
            mine = self._local_epochs[0]
            got = anchored.get(mine["number"])
            if got is not None and str(got["root"]) == mine["root"] \
                    and str(got["epoch_id"]) == mine["epoch_id"]:
                self._local_epochs.pop(0)
                continue
            logger.warning(
                "federated close: epoch %d was anchored as %s, not as this "
                "daemon closed it (%s); rewinding %d local epoch(s) onto the "
                "anchored state", mine["number"],
                (got or {}).get("epoch_id", "?"), mine["epoch_id"],
                len(self._local_epochs))
            self.rewinds += 1
            restore = getattr(self.gossip, "restore_epoch_batches", None)
            for e in self._local_epochs:
                if restore is not None and e.get("members"):
                    restore(e["members"])
            self._local_epochs = []
            self._rewind_pending = True
            break
        self._save_local_epochs()

    def _install_anchored_state(self, anchor_count: int,
                                anchored: Dict[int, Dict[str, Any]]) -> bool:
        """Install the carry-over and canonical world of the latest anchor
        (hash-verified against the chain), as the base of the next close.
        Returns False while they are not retrievable."""
        if anchor_count == 0:
            self.install_carry_over({k: {} for k in CARRY_KEYS})
            if self.canonical_tracker is not None:
                from .state_sync import CanonicalWorldTracker
                t = self.canonical_tracker
                self.canonical_tracker = CanonicalWorldTracker(
                    bandwidth=t.bandwidth, embedding_dim=t.embedding_dim,
                    equilibrate_rounds=t.equilibrate_rounds,
                    equilibrate_tolerance=t.equilibrate_tolerance)
                self._tracker_restored = True
            self._rewind_pending = False
            return True
        head = anchored.get(anchor_count) or {}
        if self.carry_source is None:
            logger.warning("federated close: no carry-over source to rewind with")
            return False
        try:
            fetch = self.carry_source()
        except Exception as e:
            logger.warning("federated close: anchored carry-over fetch failed: %s", e)
            return False
        if getattr(fetch, "status", "") != "ok" or \
                str(getattr(fetch, "epoch_id", "")) != str(head.get("epoch_id")):
            logger.info("federated close: anchored carry-over for epoch %s not "
                        "retrievable yet; waiting", head.get("epoch_id"))
            return False
        if self.canonical_tracker is not None:
            try:
                world = self.world_source() if self.world_source else None
            except Exception as e:
                logger.warning("federated close: anchored world fetch failed: %s", e)
                world = None
            if world is None or str(getattr(world, "epoch_root_hex", "")) != \
                    str(head.get("root")):
                logger.info("federated close: anchored world for epoch %s not "
                            "retrievable yet; waiting", head.get("epoch_id"))
                return False
            self._install_world(world)
        self.install_carry_over(fetch.carry)
        self._rewind_pending = False
        logger.info("federated close: installed the anchored state of epoch %s",
                    head.get("epoch_id"))
        return True

    def _pin_for_next(self, anchor_count: int,
                      anchored: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        number = anchor_count + len(self._local_epochs) + 1
        cutoffs = [int(anchored[k]["cutoff"]) for k in sorted(anchored)
                   if k <= anchor_count]
        cutoffs += [int(e["cutoff"]) for e in self._local_epochs]
        if self._local_epochs:
            prev_root = self._local_epochs[-1]["root"]
        elif anchor_count:
            prev_root = str(anchored[anchor_count]["root"])
        else:
            prev_root = "00" * 32
        return {"number": number, "cutoffs": cutoffs, "prev_root": prev_root}

    def _refresh_voice_for_anchor(self) -> None:
        """Refresh only when a new anchor landed (or the last one failed):
        every input is pinned to the latest anchor."""
        src = self.anchor_count_source
        if (src is not None and self._voice_refresh_error is None
                and self.snapshot_anchor_count is not None):
            try:
                if int(src()) == self.snapshot_anchor_count:
                    return
            except Exception as e:
                logger.debug("anchorCount read failed, refreshing: %s", e)
        self._refresh_voice()

    def _select_window(self) -> Optional[tuple]:
        """Pick this epoch's batches, or None when it is not time.

        Members are the buffered batches signed after the previous cutoff
        (``start``) and at or before this one. The cutoff is base + length;
        if no pending batch is signed by then, it slides forward in grace
        steps to the first slot that holds one. The daemon closes once its
        clock passes cutoff + grace, and closes nothing while the epoch it
        last closed is still unanchored. Returns (members, epoch_id)."""
        from .voice_state import make_epoch_id

        win = self.epoch_window
        number = int(win["number"])
        if self.chain_source is None and self._closed_epoch_number is not None and \
                self._closed_epoch_number >= number:
            logger.debug("federated close: epoch %d in flight, waiting for "
                         "its anchor", self._closed_epoch_number)
            return None
        start = win.get("start")
        pending = self.gossip.pending_epoch_batches()
        live = [b for b in pending
                if start is None or float(b.timestamp) > float(start)]
        if not live:
            return None
        grace = int(win["grace"])
        step = max(1, grace)
        cutoff = int(win["base"]) + int(win["length"])
        if int(win.get("target_share") or 0) > 0 and self.burned_source is not None:
            # Volume ends the epoch (Decision 2026-10-27). Read only time
            # one grace old, so every daemon sees the same final burns.
            from .voice_state import volume_cutoff
            try:
                vol = volume_cutoff(
                    self.burned_source, start=int(win["base"]),
                    floor=int(win["floor"]), ceiling=int(win["ceiling"]),
                    target_share_ppm=int(win["target_share"]),
                    supply=int(win.get("supply") or 0),
                    final_ts=int(float(self.clock())) - grace)
            except Exception as e:
                logger.warning("federated close: burned-volume read failed: %s", e)
                return None
            if vol is None:
                return None
            cutoff = int(vol)
        earliest = min(float(b.timestamp) for b in live)
        if earliest > cutoff:
            cutoff += step * math.ceil((earliest - cutoff) / step)
        if float(self.clock()) < cutoff + grace:
            return None
        members = [b for b in live if float(b.timestamp) <= cutoff]
        self.gossip.take_epoch_batches(members, drop_at_or_before=cutoff)
        epoch_id = make_epoch_id(number, str(win["prev_root"]), cutoff)
        if self.chain_source is None:
            self._closed_epoch_number = number
            self._save_in_flight(number, epoch_id)
        return members, epoch_id

    def _restore_canonical_tracker(self) -> bool:
        """Load the persisted checkpoint into the tracker. Returns True
        when one was restored."""
        path = self._canonical_world_path
        if self.canonical_tracker is None or path is None or not path.exists():
            return False
        from .state_sync import CanonicalWorldTracker, decode_world_checkpoint
        try:
            blob = path.read_bytes()
            checkpoint = decode_world_checkpoint(blob)
            tracker = self.canonical_tracker
            self.canonical_tracker = CanonicalWorldTracker.from_checkpoint(
                checkpoint, cid_for_blob(blob),
                bandwidth=tracker.bandwidth,
                embedding_dim=tracker.embedding_dim,
                equilibrate_rounds=tracker.equilibrate_rounds,
                equilibrate_tolerance=tracker.equilibrate_tolerance,
            )
        except Exception as e:
            logger.error(
                "canonical world checkpoint unreadable (%s); this daemon "
                "will not close until it fetches the anchored one", e)
            return False
        logger.info("canonical world tracker restored from %s (epoch %s)",
                    path.name, checkpoint.get("epoch_id"))
        return True

    def _save_canonical_world(self, blob: bytes) -> None:
        path = self._canonical_world_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(blob)
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist canonical world: %s", e)

    def _maybe_join_world(self, prev_root: Optional[str]) -> None:
        """Install the anchored canonical world when the tracker has none
        past genesis, under the same rule as the carry-over: the anchor
        must be the epoch observed immediately before this close."""
        if (self.canonical_tracker is None or self._tracker_restored
                or self._chain_at_genesis is not False
                or self.world_source is None or prev_root is None):
            return
        try:
            result = self.world_source()
        except Exception as e:
            logger.warning(
                "federated close: anchored world fetch failed (observing): %s", e)
            return
        if result is None:
            return
        if str(getattr(result, "epoch_root_hex", "") or "") != prev_root:
            logger.warning(
                "federated close: latest anchored world (epoch %s) is not the "
                "epoch observed just before this close; observing",
                getattr(result, "epoch_id", "?"))
            return
        self._install_world(result)
        logger.info("federated close: installed the anchored canonical world "
                    "(epoch %s)", result.epoch_id)

    def _install_world(self, result: Any) -> None:
        """Make an anchored, hash-verified world checkpoint the tracker's."""
        from .state_sync import CanonicalWorldTracker, encode_world_checkpoint
        tracker = self.canonical_tracker
        self.canonical_tracker = CanonicalWorldTracker(
            world=result.world, prev_world_cid=result.world_cid,
            bandwidth=tracker.bandwidth, embedding_dim=tracker.embedding_dim,
            equilibrate_rounds=tracker.equilibrate_rounds,
            equilibrate_tolerance=tracker.equilibrate_tolerance,
        )
        blob = encode_world_checkpoint(
            result.world, epoch_id=result.epoch_id,
            epoch_root_hex=result.epoch_root_hex,
            prev_world_cid=result.prev_world_cid)
        if cid_for_blob(blob) == result.world_cid:
            self._save_canonical_world(blob)
        self._tracker_restored = True
        self._tracker_failed = None

    def carry_over_paths(self) -> Dict[str, Optional[Path]]:
        """The on-disk carry-over files every close reads as INPUTS."""
        return {
            "tool_registrations": self._tool_registrations_path,
            "tool_vetting": self._tool_vetting_path,
            "tool_positions": self._tool_positions_path,
        }

    def carry_over_maps(self) -> Dict[str, Any]:
        """The in-memory carry-over (keyed like ``CARRY_KEYS``)."""
        return {
            "tool_registrations": self._tool_registrations,
            "tool_vetting": self._tool_vetting,
            "tool_positions": self._tool_positions,
        }

    def install_carry_over(self, carry: Dict[str, Any]) -> None:
        """Install a VERIFIED carry-over bundle (all three maps) as this
        daemon's close inputs: persist every file, then reload the
        in-memory maps through the normal loaders so the joiner holds
        exactly what an always-on daemon would after a restart.

        Callers must only pass maps from ``fetch_carry_over_from_chain``
        (hash-checked against the anchored payload's ``carry_cid``).
        """
        missing = [k for k in CARRY_KEYS if not isinstance(carry.get(k), dict)]
        if missing:
            raise ValueError(
                "carry-over install missing maps: " + ", ".join(missing))
        self._tool_registrations = dict(carry["tool_registrations"])
        self._save_tool_registrations()
        self._tool_vetting = dict(carry["tool_vetting"])
        self._save_tool_vetting()
        self._tool_positions = dict(carry["tool_positions"])
        self._save_tool_positions()
        self._tool_registrations = self._load_tool_registrations()
        self._tool_vetting = self._load_tool_vetting()
        self._tool_positions = self._load_tool_positions()

    def seed_marker_path(self) -> Optional[Path]:
        """Marker recording the installed genesis seed (its carry_cid),
        next to the carry files. Its presence means the seed anchor's
        bundle was installed here once and must not overwrite later
        closes' output."""
        p = self._tool_registrations_path
        return None if p is None else p.parent / "carry_seed_installed.json"

    def installed_seed_cid(self) -> Optional[str]:
        path = self.seed_marker_path()
        if path is None or not path.exists():
            return None
        try:
            return str(json.loads(path.read_text(encoding="utf-8"))
                       .get("carry_cid", "")) or None
        except (OSError, json.JSONDecodeError, AttributeError):
            return None

    def _write_seed_marker(self, carry_cid: str, epoch_id: str) -> None:
        path = self.seed_marker_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(
            {"carry_cid": str(carry_cid), "epoch_id": str(epoch_id)},
            sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _carry_files_missing(self) -> bool:
        paths = self.carry_over_paths()
        return any(v is not None and not v.exists() for v in paths.values())

    def _maybe_join_from_chain(self, prev_root: Optional[str]) -> Optional[str]:
        """Joiner bootstrap: while carry-over files are missing and the
        chain has anchors, fetch the latest anchored carry-over bundle
        (payload ``carry_cid``, sha256-verified) and install it, but only
        if that anchor is the epoch this daemon observed immediately
        before the close it is about to run (``prev_root``). A lagging
        anchor (the winner failed to anchor the last epoch) or a partially
        observed prior epoch means the bundle would be stale: observe.

        Genesis seed (docs/multi_jurisdiction.md decision 9): when the
        latest anchor is a seed at anchor index 0, its bundle is an adopted
        snapshot no daemon observed, so it installs WITHOUT the prev_root
        match and even over existing carry files (they can only hold
        pre-seed, never-anchored state). It installs exactly once: a marker
        with its carry_cid is persisted next to the carry files.

        Returns the epoch_id whose close produced the installed bundle,
        or None when nothing was installed (the daemon keeps observing).
        """
        self._seed_blocker = None
        if self.carry_source is None or self._chain_at_genesis is not False:
            return None
        paths = self.carry_over_paths()
        if any(v is None for v in paths.values()):
            return None
        files_missing = self._carry_files_missing()
        # A seed can only be the latest anchor while anchorCount == 1
        # (None = count unknown, check anyway).
        seed_window = (
            self.snapshot_anchor_count in (None, 1)
            and (files_missing or self.installed_seed_cid() is None)
        )
        if not files_missing and not seed_window:
            return None
        if files_missing and prev_root is None and not seed_window:
            logger.info(
                "federated close: joiner has not observed a prior epoch yet; "
                "observing this one before fetching the anchored carry-over")
            return None
        unknown_seed = (
            "latest anchor may be an uninstalled genesis seed (anchor 0) "
            "whose snapshot could not be fetched; not closing on pre-seed "
            "local state")
        try:
            fetch = self.carry_source()
        except ValueError as e:
            logger.error(
                "federated close: REJECTED anchored carry-over bundle "
                "(integrity check failed, staying in observe mode): %s", e)
            if seed_window:
                self._seed_blocker = unknown_seed
            return None
        except Exception as e:
            logger.warning(
                "federated close: carry-over fetch failed (observing): %s", e)
            if seed_window:
                self._seed_blocker = unknown_seed
            return None
        status = getattr(fetch, "status", "")
        is_seed = (getattr(fetch, "seed", False) is True
                   and getattr(fetch, "anchor_index", -1) == 0)
        if is_seed and status == "ok":
            if not seed_window:
                return None   # already installed here
            self.install_carry_over(fetch.carry)
            self._write_seed_marker(fetch.carry_cid, fetch.epoch_id)
            logger.info(
                "federated close: installed the genesis seed snapshot from "
                "anchor 0 (epoch %s, carry_cid=%s)",
                fetch.epoch_id, str(fetch.carry_cid)[:16])
            return str(fetch.epoch_id)
        if seed_window and status != "ok" and (
                is_seed or status != "pre_schema"):
            # The seed's bundle (or the payload that would say whether
            # anchor 0 is a seed) is not retrievable yet.
            self._seed_blocker = unknown_seed
        if not files_missing:
            return None
        if prev_root is None:
            logger.info(
                "federated close: joiner has not observed a prior epoch yet; "
                "observing this one before fetching the anchored carry-over")
            return None
        if status == "ok" and str(
                getattr(fetch, "epoch_root_hex", "") or "") != prev_root:
            logger.warning(
                "federated close: latest anchor (epoch %s, root %s) is not "
                "the epoch observed immediately before this close (root "
                "%s): the anchor lags (last winner failed to anchor) or this "
                "daemon saw only part of that epoch; refusing to install a "
                "stale carry-over, observing",
                fetch.epoch_id, str(fetch.epoch_root_hex)[:16],
                prev_root[:16])
            return None
        if status == "ok":
            self.install_carry_over(fetch.carry)
            logger.info(
                "federated close: installed verified carry-over from the "
                "anchor for epoch %s (carry_cid=%s)",
                fetch.epoch_id, str(fetch.carry_cid)[:16])
            return str(fetch.epoch_id)
        if status == "pre_schema":
            logger.warning(
                "federated close: latest anchor (epoch %s, payload schema "
                "%s) predates carry-over commitments (no carry_cid); this "
                "daemon cannot verify a carry-over and will not close until "
                "the network anchors a schema-4 epoch",
                fetch.epoch_id, fetch.payload_schema)
        elif status == "unavailable":
            logger.info(
                "federated close: anchored carry-over for epoch %s not "
                "retrievable from peers yet; observing", fetch.epoch_id)
        return None

    def participation_blockers(self) -> List[str]:
        """Reasons this daemon must not take part in a federated close.

        Empty list = clear to close. A close needs (a) chain read access,
        since voice weights and the fees-only pool are close inputs (see
        AutonetService._init_federated_close_driver), and (b) the
        carry-over files. Missing carry-over is only correct at genesis
        (chain has no anchors yet); after that, closing against an empty
        prior while peers carry the accumulated one is a deterministic
        fork.
        """
        blockers: List[str] = []
        if self.voice_source is None:
            blockers.append(
                "chain RPC not configured (substrate_address + rpc_url)")
        paths = self.carry_over_paths()
        unset = sorted(k for k, v in paths.items() if v is None)
        if unset:
            blockers.append(
                "no state dir for carry-over: " + ", ".join(unset))
        missing = sorted(
            k for k, v in paths.items() if v is not None and not v.exists())
        if missing and self._chain_at_genesis is not True:
            why = ("chain has prior anchors"
                   if self._chain_at_genesis is False
                   else "genesis status unknown")
            blockers.append(
                f"missing carry-over files ({why}): " + ", ".join(missing))
        if self._seed_blocker:
            blockers.append(self._seed_blocker)
        if self._voice_refresh_error:
            blockers.append(
                "voice-state refresh failed: " + self._voice_refresh_error)
        if self.canonical_tracker is not None:
            if self._tracker_failed:
                blockers.append(
                    "canonical world tracker failed: " + self._tracker_failed)
            elif (not self._tracker_restored
                  and self._chain_at_genesis is not True):
                why = ("chain has prior anchors"
                       if self._chain_at_genesis is False
                       else "genesis status unknown")
                blockers.append(f"no canonical world checkpoint ({why})")
        return blockers

    def _refresh_voice(self, pin: Optional[Dict[str, Any]] = None) -> None:
        """Read this close's chain inputs. All-or-nothing: on any failure
        nothing is replaced and the close is blocked (Decision 2026-10-26),
        since the previous snapshot is not what peers price against."""
        if self.voice_source is None:
            return
        try:
            state = (self.voice_source(pin=pin) if pin is not None
                     else self.voice_source()) or {}
            updates: Dict[str, Any] = {}
            if "snapshot_block" in state:
                # read_voice_state reports snapshot_block=None exactly
                # when anchorCount == 0.
                updates["_chain_at_genesis"] = (
                    state.get("snapshot_block") is None)
            owner_map = state.get("owner_map")
            weights = state.get("voice_weights")
            if isinstance(owner_map, dict):
                updates["agent_owner_map"] = {
                    str(k): str(v) for k, v in owner_map.items()}
            if isinstance(weights, dict):
                updates["voice_weights"] = {
                    str(k): float(v) for k, v in weights.items()}
            # Raw rep shares for drift weight (same read, no extra call).
            shares = state.get("rep_shares")
            if isinstance(shares, dict):
                updates["rep_shares"] = {
                    str(k): float(v) for k, v in shares.items()}
            pool = state.get("emission_pool")
            if pool is not None:
                updates["emission_pool"] = float(pool)
            if state.get("snapshot_anchor_count") is not None:
                updates["snapshot_anchor_count"] = int(
                    state["snapshot_anchor_count"])
            reviews = state.get("review_snapshot")
            if isinstance(reviews, dict):
                updates["review_snapshot"] = reviews
            if state.get("rep_supply") is not None:
                updates["rep_supply"] = int(state["rep_supply"])
            if "charter_text" in state:
                text = state.get("charter_text")
                updates["pinned_charter_text"] = str(text) if text else None
            houses = state.get("sanctioned_households")
            if isinstance(houses, list):
                updates["sanctioned_households"] = sorted(
                    str(h).lower() for h in houses)
            window = state.get("epoch_window")
            if isinstance(window, dict):
                updates["epoch_window"] = dict(window)
        except Exception as e:
            self._voice_refresh_error = str(e) or type(e).__name__
            logger.warning(
                "voice-state refresh failed (this close is blocked): %s", e)
            return
        for name, value in updates.items():
            setattr(self, name, value)
        self._voice_refresh_error = None
        # Refuse batches signed further from now than the governed skew.
        if self.epoch_window is not None and hasattr(
                self.gossip, "max_skew_seconds"):
            self.gossip.max_skew_seconds = float(self.epoch_window["skew"])

    def security_standing(self, digest: str) -> Optional[Dict[str, Any]]:
        """A tool's settled harm at the pinned snapshot, for containment
        tiers: ``{harm_sum, harm_mass, supply}`` or None when unknown. The
        same pinned read on every daemon, so tiers agree network-wide."""
        snap = self.review_snapshot or {}
        tool = (snap.get("tools") or {}).get(str(digest).lower().removeprefix("0x"))
        if tool is None:
            tool = (snap.get("tools") or {}).get(str(digest))
        if not tool:
            return None
        harm = tool.get("harm") or {}
        return {"harm_sum": int(harm.get("sum") or 0),
                "harm_mass": int(harm.get("mass") or 0),
                "supply": int(self.rep_supply or 0)}

    def _charter_changed(self) -> bool:
        """True when the jurisdiction's pinned charter differs from the
        charter this process runs. The close then refuses to run with the
        old axes and signals restart-required: every daemon reads the same
        pinned value at the same epoch, so they restart onto it together."""
        if self.charter_restart_required is not None:
            return True
        text = self.pinned_charter_text
        if not text:
            return False
        from .world_model_substrate.charter import active_charter

        active = active_charter()
        if text == active.text:
            return False
        import hashlib

        pinned_hash = "0x" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        logger.error(
            "CHARTER CHANGED: the jurisdiction's charter pinned to the "
            "previous anchor (hash=%s) differs from the active charter "
            "(hash=%s). Refusing to close on the old axes; this daemon must "
            "restart onto the new charter.", pinned_hash, active.hash,
        )
        self.charter_restart_required = text
        cb = self.on_charter_change
        if cb is not None:
            try:
                cb(text)
            except Exception:
                logger.exception("on_charter_change callback failed")
        return True

    def run(self, local_close_result: Dict[str, Any]) -> Optional[FederatedCloseResult]:
        """Drive one federated close given the local close's result.

        Returns None when there's nothing meaningful to close (no
        batches in the gossip buffer). Otherwise returns the
        deterministic federated result + which sender should anchor.
        """
        if self.voice_source is not None and self.chain_source is not None:
            return self._run_pipelined(local_close_result)
        if self.voice_source is not None:
            self._refresh_voice_for_anchor()
        if self.voice_source is not None and self.epoch_window is not None:
            if self._voice_refresh_error:
                logger.warning(
                    "federated close: voice-state refresh failed, not "
                    "closing: %s", self._voice_refresh_error)
                return None
            picked = self._select_window()
            if picked is None:
                return None
            batches, epoch_id = picked
        else:
            batches = self.gossip.drain_epoch_batches()
            if not batches:
                logger.debug("federated close: no batches buffered, skipping")
                return None
            # No chain window (library/local use): the local epoch id.
            epoch_id = str(local_close_result.get("epoch_id") or "")

        return self._close(batches, epoch_id, local_close_result)

    def _close(self, batches: List[EventBatch], epoch_id: str,
               local_close_result: Dict[str, Any], *,
               join_from_chain: bool = True) -> Optional[FederatedCloseResult]:
        """Close one epoch over ``batches`` (shared by both modes)."""
        if self._charter_changed():
            return None

        canonical = canonical_order(batches)
        if not canonical.ordered_batches:
            logger.debug(
                "federated close: canonical empty after dropping invalid senders",
            )
            return None
        # Remember this epoch's canonical root (closed or only observed)
        # so the NEXT close can check a fetched bundle is exactly the
        # output of the epoch before it.
        prev_root = self._last_observed_root
        self._last_observed_root = canonical.epoch_root().hex()

        if self.enforce_preconditions:
            if join_from_chain:
                self._maybe_join_from_chain(prev_root)
                self._maybe_join_world(prev_root)
            blockers = self.participation_blockers()
            if blockers:
                # Batches are already drained, so they don't leak into
                # the next epoch's close.
                hint = ""
                if any(b.startswith("missing carry-over") for b in blockers):
                    hint = (" (observing: the carry-over is installed from the "
                            "latest anchor's carry_cid once that anchor is the "
                            "epoch observed just before a close and its bundle "
                            "is retrievable from peers)"
                            if self.carry_source is not None else
                            " (no carry-over source wired: needs chain read "
                            "access and a blob resolver)")
                logger.warning(
                    "federated close: refusing to participate in epoch %s: %s%s",
                    local_close_result.get("epoch_id"), "; ".join(blockers),
                    hint,
                )
                return None

        # Replay the canonical sequence on a fresh charter world. This
        # produces the bit-identical authoritative_payload across
        # daemons.
        try:
            close_result = federated_epoch_close(
                canonical,
                bandwidth=self.bandwidth,
                embedding_dim=self.embedding_dim,
                pricing=self.pricing,
                tool_registrations=dict(self._tool_registrations),
                agent_owner_map=dict(self.agent_owner_map),
                voice_weights=(dict(self.voice_weights)
                               if self.voice_weights else None),
                emission_pool=self.emission_pool,
                tool_vetting=dict(self._tool_vetting),
                tool_positions=dict(self._tool_positions),
                rep_shares=(dict(self.rep_shares)
                            if self.rep_shares else None),
                review_snapshot=(dict(self.review_snapshot)
                                 if self.review_snapshot else None),
                sanctioned_households=list(self.sanctioned_households),
            )
        except Exception as e:
            logger.error(
                "federated_epoch_close failed: %s", e, exc_info=True,
            )
            return None

        # MONEY-ONLY CONTRACT SEAM. ``close_result`` carries ``agent_mint``,
        # keyed by household claimant (schema 6, Decision 2026-10-27), and
        # the ``authoritative_payload`` commits it on a 2-field
        # (claimant, amount) merkle leaf. The household submitter
        # (nodes/common/authoritative_submitter.py) reads it from the
        # anchored blob and calls ``claimFor(claimant, amount, epochIdHash,
        # proof)`` with the daemon operator key. REP is claimed DAO-side on
        # ratified ATN earnings. This driver does not itself call the chain.

        # Advance + persist the tool-registration carry-over for the
        # next epoch (returned map = carried ∪ this epoch's canonical
        # registration events, first-registration-wins).
        self._tool_registrations = dict(
            close_result.get("tool_registrations") or {}
        )
        self._save_tool_registrations()
        self._tool_vetting = dict(close_result.get("tool_vetting") or {})
        self._save_tool_vetting()
        self._tool_positions = dict(close_result.get("tool_positions") or {})
        self._save_tool_positions()
        # These are the AUTHORITATIVE positions: the ToolBond ratings pinned
        # to the previous anchor (review_snapshot above), which the daemon's
        # own local projection does not read. Push them onto the live world
        # so library ranking (infer_artifacts reads the head off the
        # observation coords) matches consensus.
        try:
            world_service = getattr(self.gossip, "world_service", None)
            applier = getattr(
                world_service, "apply_federated_tool_positions", None)
            if applier is not None and self._tool_positions:
                applier(self._tool_positions)
        except Exception as e:
            logger.warning("applying federated tool positions failed: %s", e)

        # The agreed epoch id (Decision 2026-10-26): every daemon closing
        # this window names it the same, so they build the same proposal.
        close_result["epoch_id"] = epoch_id

        # Verifiable carry-over (payload schema 4): commit the three maps
        # this close produced (the next close's inputs) as one canonical
        # bundle whose cid rides the payload. Pure function of the close
        # output, so identical on every honest daemon.
        carry_blob = encode_carry_bundle(
            self.carry_over_maps(), epoch_id=epoch_id)
        carry_cid = cid_for_blob(carry_blob)
        _payload = close_result.get("authoritative_payload")
        if _payload is not None:
            _payload["carry_cid"] = carry_cid

        # State sync: advance the cumulative canonical world and embed
        # its checkpoint cid in the payload BEFORE the payload gets
        # encoded/anchored. Deterministic across daemons, so the
        # payload stays consensus-identical.
        world_cid = ""
        world_blob = b""
        if self.canonical_tracker is not None:
            try:
                ckpt = self.canonical_tracker.on_close(
                    epoch_id,
                    str(close_result.get("epoch_root", "")),
                    [list(b.events or []) for b in canonical.ordered_batches],
                )
                world_cid = ckpt.cid
                world_blob = ckpt.blob
                payload = close_result.get("authoritative_payload")
                if payload is not None:
                    payload["world_cid"] = world_cid
                self._save_canonical_world(world_blob)
                self._tracker_restored = True
            except Exception as e:
                # The tracker's world is now half-advanced and this
                # payload has no world_cid: neither matches any peer.
                self._tracker_failed = str(e) or type(e).__name__
                self._tracker_restored = False
                logger.error(
                    "canonical world tracker failed (closes blocked until "
                    "the anchored world is fetched): %s", e, exc_info=True,
                )
                return None

        senders = self.gossip.known_senders()
        canonical_root = canonical.epoch_root() if hasattr(canonical, "epoch_root") else b""
        winner = pick_submitter(epoch_id, senders, canonical_root)
        is_winner = (winner == self.gossip.sender_pubkey) if winner else False

        logger.info(
            "federated close: epoch=%s batches=%d senders=%d winner=%s is_us=%s",
            epoch_id,
            len(canonical.ordered_batches),
            len(senders),
            winner.hex()[:16] if winner else "none",
            is_winner,
        )

        return FederatedCloseResult(
            epoch_id=epoch_id,
            close_result=close_result,
            senders=senders,
            winner=winner or b"",
            is_winner=is_winner,
            n_batches=len(canonical.ordered_batches),
            world_cid=world_cid,
            world_checkpoint_blob=world_blob,
            carry_cid=carry_cid,
            carry_bundle_blob=carry_blob,
            snapshot_anchor_count=self.snapshot_anchor_count,
        )

    def _run_pipelined(self, local_close_result: Dict[str, Any]
                       ) -> Optional[FederatedCloseResult]:
        """One tick of the pipelined mode: reconcile with the chain, rewind
        if an epoch was anchored differently, then close the next epoch
        once its cutoff plus grace has passed."""
        try:
            head = self.chain_source() or {}
            anchor_count = int(head.get("anchor_count", 0))
            anchored = {int(a["number"]): a for a in head.get("anchors", [])}
        except Exception as e:
            logger.warning("federated close: chain read failed, not closing: %s", e)
            return None
        self._chain_at_genesis = anchor_count == 0
        self._reconcile(anchor_count, anchored)
        if anchor_count and not self._local_epochs and self._carry_files_missing():
            self._rewind_pending = True   # a joiner: start from the chain
        if self._rewind_pending and not self._install_anchored_state(
                anchor_count, anchored):
            return None
        pin = self._pin_for_next(anchor_count, anchored)
        key = (pin["number"], tuple(pin["cutoffs"][-1:]), pin["prev_root"])
        if key != self._pin_key or self._voice_refresh_error:
            self._refresh_voice(pin=pin)
            if self._voice_refresh_error:
                logger.warning("federated close: voice-state refresh failed, "
                               "not closing: %s", self._voice_refresh_error)
                return None
            self._pin_key = key
        if self.epoch_window is None:
            return None
        picked = self._select_window()
        if picked is None:
            return None
        batches, epoch_id = picked
        fed = self._close(batches, epoch_id, local_close_result,
                          join_from_chain=False)
        if fed is None:
            # Nothing advanced: the batches wait for the next attempt.
            restore = getattr(self.gossip, "restore_epoch_batches", None)
            if restore is not None:
                restore(batches)
            return None
        fed.epoch_number = int(pin["number"])
        fed.parent_root = str(pin["prev_root"])
        self._local_epochs.append({
            "number": int(pin["number"]), "epoch_id": epoch_id,
            "cutoff": int(epoch_id.rsplit("-", 1)[1]),
            "root": str(fed.close_result.get("epoch_root", "")),
            "members": list(batches)})
        self._save_local_epochs()
        return fed

    # ---- tool-registration carry-over persistence -------------------

    def _load_tool_registrations(self) -> Dict[str, Dict[str, str]]:
        path = self._tool_registrations_path
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items()
                        if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("tool registrations cache unreadable (%s); "
                           "starting empty — it rebuilds from canonical "
                           "events", e)
        return {}

    def _save_tool_registrations(self) -> None:
        path = self._tool_registrations_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(dict(sorted(self._tool_registrations.items()))),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist tool registrations: %s", e)

    def _load_tool_vetting(self) -> Dict[str, Any]:
        path = self._tool_vetting_path
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("tool vetting cache unreadable (%s); starting "
                           "empty — it rebuilds from canonical events", e)
        return {}

    def _save_tool_vetting(self) -> None:
        path = self._tool_vetting_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._tool_vetting, sort_keys=True),
                           encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist tool vetting state: %s", e)

    def _load_tool_positions(self) -> Dict[str, Dict[str, Any]]:
        path = self._tool_positions_path
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items()
                        if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("tool positions cache unreadable (%s); starting "
                           "empty — it rebuilds from canonical events", e)
        return {}

    def _save_tool_positions(self) -> None:
        path = self._tool_positions_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(dict(sorted(self._tool_positions.items()))),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist tool positions: %s", e)
