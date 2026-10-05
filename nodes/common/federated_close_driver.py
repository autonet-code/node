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
import os
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
    # same canonical bundle of the five tool maps this close produced
    # (the next close's inputs); its cid rides the payload's carry_cid
    # field and every daemon publishes the blob, so a joiner can fetch
    # and verify it against the anchor instead of trusting a peer.
    carry_cid: str = ""
    carry_bundle_blob: bytes = b""


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
        tool_credibility_path: Optional[Any] = None,
        tool_review_book_path: Optional[Any] = None,
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
        # digest -> {"head": [6], "mass": [6]} — the mint-weighted review
        # centroid state. Same contract as registrations/vetting: derived
        # purely from canonical events, the on-disk copy is a rebuildable
        # cache, identical on every honest daemon.
        self._tool_positions_path: Optional[Path] = (
            Path(tool_positions_path) if tool_positions_path else None
        )
        self._tool_positions: Dict[str, Dict[str, Any]] = (
            self._load_tool_positions()
        )
        # v4.1 gradient-trust carry-over (memory/tool_economy_v4_gradient_trust.md):
        # household -> credibility multiplier (drift-weight), and the
        # per-household review book (digest -> household -> axis -> {sum,n}).
        # Same rebuildable-cache contract as positions: derived purely from
        # canonical events + carried state, identical on every honest daemon.
        self._tool_credibility_path: Optional[Path] = (
            Path(tool_credibility_path) if tool_credibility_path else None
        )
        self._tool_credibility: Dict[str, float] = (
            self._load_tool_credibility()
        )
        self._tool_review_book_path: Optional[Path] = (
            Path(tool_review_book_path) if tool_review_book_path else None
        )
        self._tool_review_book: Dict[str, Any] = self._load_tool_review_book()
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

    def carry_over_paths(self) -> Dict[str, Optional[Path]]:
        """The on-disk carry-over files every close reads as INPUTS."""
        return {
            "tool_registrations": self._tool_registrations_path,
            "tool_vetting": self._tool_vetting_path,
            "tool_positions": self._tool_positions_path,
            "tool_credibility": self._tool_credibility_path,
            "tool_review_book": self._tool_review_book_path,
        }

    def carry_over_maps(self) -> Dict[str, Any]:
        """The in-memory carry-over (keyed like ``CARRY_KEYS``)."""
        return {
            "tool_registrations": self._tool_registrations,
            "tool_vetting": self._tool_vetting,
            "tool_positions": self._tool_positions,
            "tool_credibility": self._tool_credibility,
            "tool_review_book": self._tool_review_book,
        }

    def install_carry_over(self, carry: Dict[str, Any]) -> None:
        """Install a VERIFIED carry-over bundle (all five maps) as this
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
        self._tool_credibility = dict(carry["tool_credibility"])
        self._save_tool_credibility()
        self._tool_review_book = dict(carry["tool_review_book"])
        self._save_tool_review_book()
        self._tool_registrations = self._load_tool_registrations()
        self._tool_vetting = self._load_tool_vetting()
        self._tool_positions = self._load_tool_positions()
        self._tool_credibility = self._load_tool_credibility()
        self._tool_review_book = self._load_tool_review_book()

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

        Returns the epoch_id whose close produced the installed bundle,
        or None when nothing was installed (the daemon keeps observing).
        """
        if self.carry_source is None or self._chain_at_genesis is not False:
            return None
        paths = self.carry_over_paths()
        if any(v is None for v in paths.values()):
            return None
        if not self._carry_files_missing():
            return None
        if prev_root is None:
            logger.info(
                "federated close: joiner has not observed a prior epoch yet; "
                "observing this one before fetching the anchored carry-over")
            return None
        try:
            fetch = self.carry_source()
        except ValueError as e:
            logger.error(
                "federated close: REJECTED anchored carry-over bundle "
                "(integrity check failed, staying in observe mode): %s", e)
            return None
        except Exception as e:
            logger.warning(
                "federated close: carry-over fetch failed (observing): %s", e)
            return None
        status = getattr(fetch, "status", "")
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
        return blockers

    def _refresh_voice(self) -> None:
        if self.voice_source is None:
            return
        try:
            state = self.voice_source() or {}
            if "snapshot_block" in state:
                # read_voice_state reports snapshot_block=None exactly
                # when anchorCount == 0.
                self._chain_at_genesis = state.get("snapshot_block") is None
            owner_map = state.get("owner_map")
            weights = state.get("voice_weights")
            if isinstance(owner_map, dict):
                self.agent_owner_map = {
                    str(k): str(v) for k, v in owner_map.items()}
            if isinstance(weights, dict):
                self.voice_weights = {
                    str(k): float(v) for k, v in weights.items()}
            # Raw rep shares for drift weight (same read, no extra call).
            shares = state.get("rep_shares")
            if isinstance(shares, dict):
                self.rep_shares = {
                    str(k): float(v) for k, v in shares.items()}
            pool = state.get("emission_pool")
            if pool is not None:
                self.emission_pool = float(pool)
        except Exception as e:
            logger.warning(
                "voice-state refresh failed (keeping previous maps): %s", e)

    def run(self, local_close_result: Dict[str, Any]) -> Optional[FederatedCloseResult]:
        """Drive one federated close given the local close's result.

        Returns None when there's nothing meaningful to close (no
        batches in the gossip buffer). Otherwise returns the
        deterministic federated result + which sender should anchor.
        """
        batches = self.gossip.drain_epoch_batches()
        if not batches:
            logger.debug("federated close: no batches buffered, skipping")
            return None

        self._refresh_voice()

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
                tool_credibility=dict(self._tool_credibility),
                tool_review_book=dict(self._tool_review_book),
            )
        except Exception as e:
            logger.error(
                "federated_epoch_close failed: %s", e, exc_info=True,
            )
            return None

        # MONEY-ONLY CONTRACT SEAM (Decision 2026-07-10). ``close_result``
        # carries ``agent_mint`` and the ``authoritative_payload`` (schema 3)
        # commits it on a 2-field (agent, amount) merkle leaf. The per-agent
        # submitter (nodes/common/authoritative_submitter.py — owned by the
        # contract agent) reads agent_mint from the anchored payload/blob and
        # calls ``record_training_for_epoch(private_key, amount,
        # epoch_id_hash, proof)`` /
        # ``recordTrainingForEpoch(amount, epochIdHash, proof)``. REP is no
        # longer minted on this path — it is claimed DAO-side (RepToken) on
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
        # These are the AUTHORITATIVE positions: this close weighted review
        # drift by rep_share × credibility (see the tool_positions=... /
        # rep_shares=... arguments above), which the daemon's own local
        # projection does not. Push them onto the live world so library
        # ranking (infer_artifacts reads the drifted head off the
        # observation coords) matches consensus instead of an unweighted
        # local view in which zero-reputation reviews still move rankings.
        try:
            world_service = getattr(self.gossip, "world_service", None)
            applier = getattr(
                world_service, "apply_federated_tool_positions", None)
            if applier is not None and self._tool_positions:
                applier(self._tool_positions)
        except Exception as e:
            logger.warning("applying federated tool positions failed: %s", e)
        # v4.1 carried gradient-trust state (rebuildable cache).
        self._tool_credibility = dict(
            close_result.get("tool_credibility") or {})
        self._save_tool_credibility()
        self._tool_review_book = dict(
            close_result.get("tool_review_book") or {})
        self._save_tool_review_book()

        # Inherit epoch_id from the local close so on-chain dedup
        # (isAnchored(epoch_id)) can reject duplicate submissions.
        epoch_id = str(local_close_result.get("epoch_id") or "")
        close_result["epoch_id"] = epoch_id

        # Verifiable carry-over (payload schema 4): commit the five maps
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
            except Exception as e:
                logger.error(
                    "canonical world tracker failed: %s", e, exc_info=True,
                )

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
        )

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

    def _load_tool_credibility(self) -> Dict[str, float]:
        path = self._tool_credibility_path
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("tool credibility cache unreadable (%s); starting "
                           "empty — it rebuilds from canonical events", e)
        return {}

    def _save_tool_credibility(self) -> None:
        path = self._tool_credibility_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(dict(sorted(self._tool_credibility.items()))),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist tool credibility: %s", e)

    def _load_tool_review_book(self) -> Dict[str, Any]:
        path = self._tool_review_book_path
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("tool review book cache unreadable (%s); starting "
                           "empty — it rebuilds from canonical events", e)
        return {}

    def _save_tool_review_book(self) -> None:
        path = self._tool_review_book_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._tool_review_book, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("failed to persist tool review book: %s", e)
