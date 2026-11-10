"""Federated-close driver — runs at epoch boundaries to produce the
deterministic authoritative result and pick the on-chain submitter.

The pieces this glues together (all already exist and tested):

  - ``EventGossip.drain_epoch_batches()`` — gather all batches that
    arrived (own + peer) in the closing epoch.
  - ``canonical_order(batches)`` — sender-grouped Merkle-rooted
    deterministic ordering. Bit-identical across honest daemons that
    received the same batch set.
  - ``federated_epoch_close(canonical, ...)`` — the review mint and the
    next carry-over from the canonical events and the pinned chain
    inputs; its ``authoritative_payload`` is bit-identical too.

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
    """Per-daemon driver invoked from the library service's epoch-close
    subscriber. Stateless across epochs — each call builds a fresh
    canonical order from the gossip's per-epoch buffer.
    """

    def __init__(
        self,
        gossip: EventGossip,
        *,
        tool_registrations_path: Optional[Any] = None,
        tool_rounds_path: Optional[Any] = None,
        tool_ratings_path: Optional[Any] = None,
    ):
        self.gossip = gossip
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
        # Zero-arg hook: does this daemon take part in consensus (has a
        # registered agent)? False => it follows the anchors and closes
        # nothing (_follow). None => always participates.
        self.participating: Optional[Callable[[], bool]] = None
        self._followed_count: Optional[int] = None
        # (from_ts, to_ts) -> ATN burned by fees between them, from
        # Substrate's on-chain checkpoints: with a target share set, an
        # epoch ends on that volume (voice_state.volume_cutoff).
        self.burned_source: Optional[Callable[[int, int], int]] = None
        # floor_ts -> batches gained: asks peers for the open epoch's batches
        # signed after floor_ts that this daemon is missing, and feeds them
        # to the gossip (issue #41). Called once per window, before closing.
        self.catchup: Optional[Callable[[float], int]] = None
        self._caught_up: Optional[tuple] = None
        self._catchup_key: Optional[tuple] = None
        self._catchup_rounds = 0
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
        # Review-round carry-over (Decision 2026-11-07): each open tool's
        # ToolBond rounds, the per-epoch review counts and the unminted
        # pool. Derived purely from prior closes' pinned snapshots, so the
        # on-disk copy is identical on every honest daemon.
        self._tool_rounds_path: Optional[Path] = (
            Path(tool_rounds_path) if tool_rounds_path else None
        )
        self._tool_rounds: Dict[str, Any] = self._load_tool_rounds()
        # Ratings carry-over (Decision 2026-11-09): digest -> the plain
        # per-field means of its ToolBond rating pinned to the previous
        # anchor (federated_reconcile.tool_rating). Derived purely from
        # prior closes' pinned snapshots; the on-disk copy is a cache,
        # identical on every honest daemon.
        self._tool_ratings_path: Optional[Path] = (
            Path(tool_ratings_path) if tool_ratings_path else None
        )
        self._tool_ratings: Dict[str, Dict[str, Any]] = (
            self._load_tool_ratings()
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
        # REP total supply at the same pin.
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
        # Breakthrough gate pinned with the voice state (None = off).
        self.mint_gate: Optional[tuple] = None
        # (review_decay, reviewer_min_rep_bps), pinned like mint_gate.
        self.review_params: tuple = (1.0, 0)
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
        # anchored_state.CarryOverFetch for the latest anchor (wired by the
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
        base = self._tool_registrations_path
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
        base = self._tool_registrations_path
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
                "federated close: epoch %d was anchored as %s root %s, not as "
                "this daemon closed it (%s root %s); rewinding %d local "
                "epoch(s) onto the anchored state", mine["number"],
                (got or {}).get("epoch_id", "?"),
                str((got or {}).get("root", "?"))[:16],
                mine["epoch_id"], str(mine["root"])[:16],
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
        """Install the carry-over of the latest anchor (hash-verified
        against the chain) as the base of the next close. Returns False
        while it is not retrievable."""
        if anchor_count == 0:
            self.install_carry_over({k: {} for k in CARRY_KEYS})
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
        # A quiet epoch closes too (Decision 2026-11-07): tool reviews settle
        # on ToolBond and pay through the close, and tool use is no longer
        # gossiped, so settlement can't wait for gossip. Epoch 1 still waits
        # for a first batch (no window start yet).
        if not live and start is None:
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
        if live:
            earliest = min(float(b.timestamp) for b in live)
            if earliest > cutoff:
                cutoff += step * math.ceil((earliest - cutoff) / step)
        if float(self.clock()) < cutoff + grace:
            return None
        # Before closing, ask peers for this epoch's batches we never saw
        # (issue #41: joined late, slept, or lost peers mid-epoch), once per
        # window; then pick again over the fuller buffer.
        floor = float(start) if start is not None else 0.0
        ckey = (number, floor)
        if self.catchup is not None and self._caught_up != ckey:
            self._caught_up = ckey
            try:
                got = int(self.catchup(floor) or 0)
            except Exception as e:  # noqa: BLE001 — close with what we have
                logger.warning("federated close: batch catch-up failed: %s", e)
                got = 0
            if got:
                logger.info("federated close: caught up %d batch(es) for epoch %d",
                            got, number)
                # A peer serves 500 at a time: ask again until nothing is new,
                # up to a limit so a peer cannot keep the close waiting.
                self._catchup_rounds = self._catchup_rounds + 1 if self._catchup_key == ckey else 1
                self._catchup_key = ckey
                if self._catchup_rounds < 20:
                    self._caught_up = None
                return self._select_window()
        members = [b for b in live if float(b.timestamp) <= cutoff]
        self.gossip.take_epoch_batches(members, drop_at_or_before=cutoff)
        epoch_id = make_epoch_id(number, str(win["prev_root"]), cutoff)
        if self.chain_source is None:
            self._closed_epoch_number = number
            self._save_in_flight(number, epoch_id)
        return members, epoch_id

    def carry_over_paths(self) -> Dict[str, Optional[Path]]:
        """The on-disk carry-over files every close reads as INPUTS."""
        return {
            "tool_registrations": self._tool_registrations_path,
            "tool_rounds": self._tool_rounds_path,
            "tool_ratings": self._tool_ratings_path,
        }

    def carry_over_maps(self) -> Dict[str, Any]:
        """The in-memory carry-over (keyed like ``CARRY_KEYS``)."""
        return {
            "tool_registrations": self._tool_registrations,
            "tool_rounds": self._tool_rounds,
            "tool_ratings": self._tool_ratings,
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
        self._tool_rounds = dict(carry["tool_rounds"])
        self._save_tool_rounds()
        self._tool_ratings = dict(carry["tool_ratings"])
        self._save_tool_ratings()
        self._tool_registrations = self._load_tool_registrations()
        self._tool_rounds = self._load_tool_rounds()
        self._tool_ratings = self._load_tool_ratings()
        # The anchored state's pruned tools (Decision 2026-11-10).
        try:
            library = getattr(self.gossip, "library", None)
            apply = getattr(library, "apply_pruned", None)
            if apply is not None:
                apply(list(self._tool_rounds.get("pruned") or []))
        except Exception as e:  # noqa: BLE001 - pruning never blocks the install
            logger.warning("applying the anchored pruned tools failed: %s", e)
        # The anchored registrations, so the library fetches and indexes
        # every registered manifest this daemon missed on gossip.
        try:
            library = getattr(self.gossip, "library", None)
            note = getattr(library, "note_registrations", None)
            if note is not None:
                note(self._tool_registrations)
        except Exception as e:  # noqa: BLE001 - discovery never blocks the install
            logger.warning("noting the anchored registrations failed: %s", e)

    def pruned_tools(self) -> List[str]:
        """The tools the latest close (or installed anchor) pruned."""
        return sorted(str(d) for d in (self._tool_rounds.get("pruned") or []))

    def tool_unpaid(self, digest: str) -> Optional[Dict[str, Any]]:
        """What the tool's author side would have paid had it not concluded
        on a malware verdict (``{"amounts": {recipient: ATN}, "on":
        [digests]}``), or None: the amounts a clearing proposal pays from
        the treasury (Decision 2026-11-10)."""
        key = str(digest).lower().removeprefix("0x")
        entry = ((self._tool_rounds.get("tools") or {}).get(key) or {}).get("unpaid")
        return dict(entry) if isinstance(entry, dict) else None

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
            if isinstance(state.get("mint_gate"), (list, tuple)) and len(state["mint_gate"]) == 2:
                updates["mint_gate"] = (int(state["mint_gate"][0]), int(state["mint_gate"][1]))
            if isinstance(state.get("review_params"), (list, tuple)) and len(state["review_params"]) == 2:
                updates["review_params"] = (float(state["review_params"][0]),
                                            int(state["review_params"][1]))
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

    def tool_rating(self, digest: str) -> Optional[Dict[str, Any]]:
        """A tool's rating as of the latest close (plain means,
        ``federated_reconcile.tool_rating``), or None when this daemon has
        none for it. The same anchored state on every daemon."""
        key = str(digest).lower().removeprefix("0x")
        entry = self._tool_ratings.get(key)
        return dict(entry) if isinstance(entry, dict) else None

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
        from .library.charter import active_charter

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
        # An empty window still closes (a quiet epoch); a window whose every
        # batch was dropped as invalid does not.
        if batches and not canonical.ordered_batches:
            logger.debug(
                "federated close: canonical empty after dropping invalid senders",
            )
            return None
        # Remember this epoch's canonical root (closed or only observed)
        # so the NEXT close can check a fetched bundle is exactly the
        # output of the epoch before it.
        prev_root = self._last_observed_root
        self._last_observed_root = canonical.epoch_root().hex()
        # What this daemon closed over, per sender: two honest daemons
        # whose roots differ can be compared line by line.
        counts: Dict[bytes, int] = {}
        for b in canonical.ordered_batches:
            counts[b.sender_pubkey] = counts.get(b.sender_pubkey, 0) + 1
        logger.info(
            "federated close inputs: epoch=%s root=%s senders=[%s] dropped=[%s]",
            epoch_id, self._last_observed_root[:16],
            ", ".join(f"{pk.hex()[:8]}:{counts.get(pk, 0)}:{r.hex()[:8]}"
                      for pk, r in sorted(canonical.per_sender_root.items())),
            ", ".join(f"{pk.hex()[:8]}:{why}"
                      for pk, why in sorted(canonical.dropped_senders.items())))

        if self.enforce_preconditions:
            if join_from_chain:
                self._maybe_join_from_chain(prev_root)
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

        # A chain-backed close past genesis weighs by REP even when the read
        # came back empty: no REP means no weight and the gate stays on,
        # never the chain-free regime's weight 1.0 for everyone.
        chain_backed = (self.voice_source is not None
                        and self._chain_at_genesis is False)

        # The close is a pure function of the canonical events, the
        # carry-over and the pinned chain inputs: the authoritative_payload
        # is bit-identical across daemons.
        try:
            close_result = federated_epoch_close(
                canonical,
                tool_registrations=dict(self._tool_registrations),
                agent_owner_map=dict(self.agent_owner_map),
                emission_pool=self.emission_pool,
                tool_rounds=dict(self._tool_rounds),
                rep_shares=(dict(self.rep_shares or {})
                            if self.rep_shares or chain_backed else None),
                review_snapshot=(dict(self.review_snapshot)
                                 if self.review_snapshot else None),
                sanctioned_households=list(self.sanctioned_households),
                mint_gate=self.mint_gate,
                review_decay=float(self.review_params[0]),
                reviewer_min_rep_bps=int(self.review_params[1]),
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
        self._tool_rounds = dict(close_result.get("tool_rounds") or {})
        self._save_tool_rounds()
        self._tool_ratings = dict(close_result.get("tool_ratings") or {})
        self._save_tool_ratings()
        # The economy graph shows this close's registrations, ratings
        # (ToolBond's, pinned to the previous anchor) and mint.
        try:
            library = getattr(self.gossip, "library", None)
            noter = getattr(library, "note_federated_close", None)
            if noter is not None:
                noter(close_result, epoch_id=epoch_id)
        except Exception as e:
            logger.warning("noting the federated close for the economy graph failed: %s", e)

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
            carry_cid=carry_cid,
            carry_bundle_blob=carry_blob,
            snapshot_anchor_count=self.snapshot_anchor_count,
        )

    def _follow(self, anchor_count: int,
                anchored: Dict[int, Dict[str, Any]]) -> None:
        """A daemon not in consensus (no registered agent) closes nothing:
        it installs each new anchor's state and drops buffered batches the
        anchors have passed. Its own unanchored closes, if any, are void."""
        if self._local_epochs:
            self._local_epochs = []
            self._save_local_epochs()
        if anchor_count != self._followed_count:
            if self._install_anchored_state(anchor_count, anchored):
                self._followed_count = anchor_count
            else:
                # Retried next tick; and if this daemon starts participating
                # first, its first close installs the anchored state.
                self._rewind_pending = True
        last = anchored.get(anchor_count)
        take = getattr(self.gossip, "take_epoch_batches", None)
        if last and last.get("cutoff") and take is not None:
            take([], drop_at_or_before=float(last["cutoff"]))
        return None

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
        if self.participating is not None and not self.participating():
            return self._follow(anchor_count, anchored)
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

    def _load_tool_rounds(self) -> Dict[str, Any]:
        path = self._tool_rounds_path
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("tool rounds cache unreadable (%s); starting "
                           "empty — it rebuilds from canonical events", e)
        return {}

    def _save_tool_rounds(self) -> None:
        path = self._tool_rounds_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._tool_rounds, sort_keys=True),
                           encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist tool rounds state: %s", e)

    def _load_tool_ratings(self) -> Dict[str, Dict[str, Any]]:
        path = self._tool_ratings_path
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items()
                        if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("tool ratings cache unreadable (%s); starting "
                           "empty: the next close rebuilds it", e)
        return {}

    def _save_tool_ratings(self) -> None:
        path = self._tool_ratings_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(dict(sorted(self._tool_ratings.items()))),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist tool ratings: %s", e)
