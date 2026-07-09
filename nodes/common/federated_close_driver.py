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
        # v4.1 gradient trust: the RAW reputation share per household
        # (rep/rep_supply, NO epsilon floor) — distinct from voice_weights
        # (which carries the floor for mint bootstrap). Drift weight and
        # the rep/ATN mint split need the un-floored share. Plus the
        # reputation total supply, the β-cap maturity clock. Both come from
        # the same voice_state read (no extra chain round-trip). Empty /
        # None => local regime (uncapped β, weight-1.0 drift).
        self.rep_shares: Dict[str, float] = {}
        self.rep_supply: Optional[float] = None
        # Fee-recycled emission pool (docs/epoch_economics.md): base
        # floor + burned service fees in the snapshot anchor's window,
        # computed by the same voice_source refresh. None (no chain
        # source) = no normalization — raw mint units, the legacy/local
        # behavior.
        self.emission_pool: Optional[float] = None
        # Optional zero-arg refresh hook returning
        # {"owner_map": {...}, "voice_weights": {...}} — wired by the
        # host when chain access exists (see AutonetService.
        # attach_chain_submission). Called at the top of each run so the
        # close prices this epoch's voices from current chain state; on
        # failure the previous maps stand (stale beats forked).
        self.voice_source: Optional[Any] = None

    def _refresh_voice(self) -> None:
        if self.voice_source is None:
            return
        try:
            state = self.voice_source() or {}
            owner_map = state.get("owner_map")
            weights = state.get("voice_weights")
            if isinstance(owner_map, dict):
                self.agent_owner_map = {
                    str(k): str(v) for k, v in owner_map.items()}
            if isinstance(weights, dict):
                self.voice_weights = {
                    str(k): float(v) for k, v in weights.items()}
            # v4.1: raw rep shares + supply (same read, no extra call).
            shares = state.get("rep_shares")
            if isinstance(shares, dict):
                self.rep_shares = {
                    str(k): float(v) for k, v in shares.items()}
            supply = state.get("rep_supply")
            if supply is None:
                supply = state.get("supply")
            if supply is not None:
                self.rep_supply = float(supply)
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
                rep_supply=self.rep_supply,
                tool_credibility=dict(self._tool_credibility),
                tool_review_book=dict(self._tool_review_book),
            )
        except Exception as e:
            logger.error(
                "federated_epoch_close failed: %s", e, exc_info=True,
            )
            return None

        # v4.1 [R6] rep/ATN split — CONTRACT SEAM. ``close_result`` now
        # carries ``agent_rep`` (decoupled per-agent reputation increment)
        # alongside ``agent_mint``, and the ``authoritative_payload`` (schema
        # 2) commits both so the anchor's mint merkle ratifies (agent,
        # agent_mint, agent_rep). Downstream, the per-agent submitter
        # (nodes/common/authoritative_submitter.py — owned by the contract
        # agent) reads agent_rep from the anchored payload/blob and passes it
        # as the ``rep_amount``/``rep_contribution`` arg to the extended
        # ``record_training_for_epoch(private_key, amount, epoch_id_hash,
        # proof, rep_amount=...)`` /
        # ``recordTrainingForEpoch(amount, repAmount, epochIdHash, proof)``.
        # The mint-root builder (mint_merkle.py, also contract-agent-owned)
        # must build leaves from (agent, agent_mint, agent_rep). This driver
        # surfaces agent_rep on FederatedCloseResult.close_result for those
        # consumers; it does not itself call the chain.

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
