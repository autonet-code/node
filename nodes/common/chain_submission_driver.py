"""Chain submission driver — anchor + household mint claims.

Triggered by ``FederatedCloseDriver`` results (Phase 10.3a). Two
responsibilities:

  - **Anchor**: every daemon encodes the identical anchor from the
    bit-identical close. ``Substrate.anchor`` is governor-only (Decision
    2026-10-16), so the anchor travels as a DAO proposal: the rotating
    winner proposes, every daemon votes For with its agents' delegated
    REP, and any daemon queues and executes (``governed_anchor``). On a dev
    network whose Substrate governor is this daemon's own key, the winner
    anchors directly.

  - **Household claims** (Decision 2026-10-27): the anchored mint is
    keyed by owner wallet. For each household this daemon serves (the
    configured owner wallet and the on-chain owners of its registered
    agents) an ``AuthoritativeChainSubmitter`` reads the anchor, fetches
    the blob, decodes the household's share and calls ``claimFor``,
    signed by the daemon operator key (``daemon_private_key``), which
    pays the gas. The ATN goes to the owner whoever submits.

The daemon operator key also anchors, proposes and votes: the owner
delegates its REP votes to it.

The anchor blob (off-chain) is written to a ``BlobResolver`` that
the submitters read from. For single-process testing an
``InMemoryBlobResolver`` is fine; cross-daemon needs a libp2p-backed
resolver (Phase 10.3d).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .authoritative_submitter import (
    AuthoritativeChainSubmitter,
    AuthoritativeSubmissionResult,
    AuthoritativeSubmitterConfig,
)
from .blob_resolver import BlobResolver, InMemoryBlobResolver
from .epoch_anchorer import (
    AnchorResult,
    EpochAnchorer,
    EpochAnchorerConfig,
)
from .federated_close_driver import FederatedCloseResult
from .anchor_election import Election, ElectionAnchorer
from .governed_anchor import AnchorProposal, GovernedAnchorer, Voter


logger = logging.getLogger(__name__)

# A pending anchor proposal is dropped after this long (the reference
# DAO's voting period is 7 days; the timelock adds minutes).
_MAX_PROPOSAL_AGE_S = 9 * 24 * 3600
_TERMINAL = {"defeated", "canceled", "expired"}


@dataclass
class ChainSubmissionConfig:
    """Per-daemon chain config. Mirrors BlockchainConfig but renamed
    so the submitter doesn't import the broader autonet config."""
    substrate_address: str
    rpc_url: str = "http://localhost:8545"
    chain_id: int = 31337
    daemon_private_key: str = ""
    # The DAO Governor anchor proposals go through. Empty => only direct
    # anchoring (dev networks whose Substrate governor is this daemon's key).
    governor_address: str = ""
    # The AnchorElection (Decision 2026-10-26). Set => anchors go through
    # single-choice elections (anchor_election.ElectionAnchorer).
    anchor_election_address: str = ""
    # The household this daemon serves (Decision 2026-10-27). Claimed for
    # alongside the on-chain owners of the local agents.
    owner_wallet: str = ""


# Resolver: local agent_id (YAML id) -> (address, private_key) or None.
# Used by the chain submission driver to find the per-agent signing
# key when constructing AuthoritativeChainSubmitters.
@dataclass
class AgentChainIdentity:
    address: str
    private_key: str


AgentChainResolver = Callable[[], List[AgentChainIdentity]]


class ChainSubmissionDriver:
    """Owns the anchor submitter and a cache of per-agent submitters.

    A single instance per daemon. Subscribes via
    ``handle_federated_close(fed_result)`` to ``FederatedCloseDriver``
    output (registered through
    ``AutonetService.add_federated_close_subscriber``).
    """

    def __init__(
        self,
        *,
        config: ChainSubmissionConfig,
        agent_chain_resolver: AgentChainResolver,
        blob_resolver: Optional[BlobResolver] = None,
        anchorer: Optional[EpochAnchorer] = None,
        governed: Optional[GovernedAnchorer] = None,
        election: Optional[ElectionAnchorer] = None,
        gossip: Optional[Any] = None,
        tick_interval: float = 15.0,
        submitter_web3: Optional[Any] = None,
        **_ignored: Any,
    ):
        self.config = config
        self._agent_resolver = agent_chain_resolver
        # Use `is None` rather than `or` because BlobResolver impls
        # define ``__len__`` and an empty resolver evaluates falsy.
        self._blob_resolver = (
            blob_resolver if blob_resolver is not None else InMemoryBlobResolver()
        )
        self._lock = threading.RLock()
        # Optional shared Web3 for the per-agent submitters (tests and
        # in-process chains); None = each submitter dials config.rpc_url.
        self._submitter_web3 = submitter_web3
        # Cache: claimant address (lowercase) -> AuthoritativeChainSubmitter.
        self._submitter_cache: Dict[str, AuthoritativeChainSubmitter] = {}
        # Anchor side. Configurable so tests can pass an in-process
        # anchorer with eth_tester.
        self._anchorer = anchorer or self._build_anchorer()
        self._governed = governed
        self._election = election
        # epoch_id -> the epoch's election (election path).
        self._elections: "OrderedDict[str, Election]" = OrderedDict()
        # Pipelined elections: epoch number -> the anchor fields this daemon
        # prepared for it, so the next epoch's option chains onto them.
        self._fields_by_number: Dict[int, Any] = {}
        # epoch_id -> pending anchor proposal (governed path).
        self.tick_interval = float(tick_interval)
        self._pending: "OrderedDict[str, AnchorProposal]" = OrderedDict()
        self._ticker: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def governed_anchorer(self) -> Optional[GovernedAnchorer]:
        if self._governed is None and self._anchorer is not None:
            try:
                contract = self._anchorer._get_contract()
                self._governed = GovernedAnchorer(
                    w3=self._anchorer._w3, substrate=contract,
                    governor_address=self.config.governor_address,
                    chain_id=self.config.chain_id,
                    daemon_key=self.config.daemon_private_key)
            except Exception as e:
                logger.warning("governed anchorer unavailable: %s", e)
                return None
        return self._governed

    def election_anchorer(self) -> Optional[ElectionAnchorer]:
        """The election path, when an AnchorElection is configured."""
        if self._election is None and self._anchorer is not None                 and self.config.anchor_election_address:
            try:
                contract = self._anchorer._get_contract()
                self._election = ElectionAnchorer(
                    w3=self._anchorer._w3, substrate=contract,
                    election_address=self.config.anchor_election_address,
                    governor_address=self.config.governor_address,
                    chain_id=self.config.chain_id,
                    daemon_key=self.config.daemon_private_key)
            except Exception as e:
                logger.warning("anchor election unavailable: %s", e)
                return None
        return self._election

    def _build_anchorer(self) -> Optional[EpochAnchorer]:
        if not self.config.substrate_address:
            logger.info(
                "ChainSubmissionDriver: no substrate_address; "
                "chain submission disabled"
            )
            return None
        if not self.config.daemon_private_key:
            logger.info(
                "ChainSubmissionDriver: no daemon_private_key; "
                "chain submission disabled"
            )
            return None
        return EpochAnchorer(
            EpochAnchorerConfig(
                epoch_anchor_address=self.config.substrate_address,
                rpc_url=self.config.rpc_url,
                chain_id=self.config.chain_id,
                private_key=self.config.daemon_private_key,
            )
        )

    @property
    def blob_resolver(self) -> BlobResolver:
        return self._blob_resolver

    @property
    def enabled(self) -> bool:
        return self._anchorer is not None

    def handle_federated_close(self, fed: FederatedCloseResult) -> Dict[str, Any]:
        """Anchor (if winner) + submit per-agent mints. Subscriber
        method registered with AutonetService.

        Returns a diagnostic dict with what happened.
        """
        out: Dict[str, Any] = {
            "epoch_id": fed.epoch_id,
            "is_winner": fed.is_winner,
            "anchored": False,
            "anchor_tx": "",
            "agent_submissions": [],
        }

        if not self.enabled:
            return out

        # State sync: every daemon computed the same canonical-world
        # checkpoint blob — publish it so rejoining peers can fetch it
        # from anyone, not just the anchor winner.
        if fed.world_checkpoint_blob:
            try:
                self._blob_resolver.put(fed.world_checkpoint_blob)
                out["world_cid"] = fed.world_cid
            except Exception as e:
                logger.warning("world checkpoint blob put failed: %s", e)
        # Verifiable carry-over (payload schema 4): same contract as the
        # world blob. Every daemon computed the identical bundle, so every
        # daemon serves it; a joiner verifies it against the anchored
        # payload's carry_cid, so it doesn't matter which peer answers.
        carry_blob = getattr(fed, "carry_bundle_blob", b"")
        if carry_blob:
            try:
                self._blob_resolver.put(carry_blob)
                out["carry_cid"] = fed.carry_cid
            except Exception as e:
                logger.warning("carry-over bundle blob put failed: %s", e)

        prepared = self._prepare(fed, out)
        if prepared is not None:
            self._publish_anchor_blobs(prepared.result)
            ea = self.election_anchorer()
            ga = None if ea is not None else self.governed_anchorer()
            if ea is not None:
                try:
                    election = ea.open(fed.epoch_id, prepared.fields.as_args())
                except Exception as e:
                    logger.warning("anchor election open failed for %s: %s",
                                   fed.epoch_id, e)
                    out["anchor_error"] = f"election open failed: {e}"
                else:
                    out["anchor_election"] = election.epoch_number
                    out["anchor_state"] = self._step_election(election)
                    if out["anchor_state"] == "executed":
                        out["anchored"] = True
                    else:
                        with self._lock:
                            self._elections[fed.epoch_id] = election
                        self._ensure_ticker()
                        return out
            elif ga is not None and ga.direct:
                if fed.is_winner:
                    anchor_result = self._anchor_direct(prepared)
                    out["anchored"] = anchor_result.success
                    out["anchor_tx"] = anchor_result.tx_hash
                    if not anchor_result.success:
                        out["anchor_error"] = anchor_result.error
            elif ga is not None and ga.available:
                try:
                    proposal = ga.build(fed.epoch_id, prepared.fields.as_args())
                except Exception as e:
                    logger.warning("anchor proposal build failed for %s: %s",
                                   fed.epoch_id, e)
                    out["anchor_error"] = f"proposal build failed: {e}"
                else:
                    out["anchor_proposal"] = hex(proposal.proposal_id)
                    out["anchor_state"] = self._step(proposal, may_propose=fed.is_winner)
                    if out["anchor_state"] == "executed":
                        out["anchored"] = True
                    else:
                        # Mints wait for the anchor; the ticker submits them
                        # once the proposal executes.
                        with self._lock:
                            self._pending[fed.epoch_id] = proposal
                        self._ensure_ticker()
                        return out
            else:
                out["anchor_error"] = ("no DAO governor configured and this "
                                       "daemon is not Substrate's governor")
                logger.warning("epoch %s: %s", fed.epoch_id, out["anchor_error"])

        # Household claims. Run regardless of is_winner: every daemon
        # claims for its households once the anchor is on chain.
        agent_results = self._submit_agents(fed.epoch_id)
        out["agent_submissions"] = agent_results
        return out

    def _prepare(self, fed: FederatedCloseResult,
                 out: Dict[str, Any]) -> Optional[Any]:
        """Prepare the anchor. None when this daemon must not anchor
        (already anchored, stale snapshot, malformed close)."""
        anchorer = self._anchorer
        try:
            if anchorer.is_anchored(fed.epoch_id):
                out["anchor_skipped"] = "already anchored"
                return None
            number = getattr(fed, "epoch_number", None)
            if number is not None:
                return self._prepare_pipelined(fed, int(number), out)
            snap = getattr(fed, "snapshot_anchor_count", None)
            if snap is not None:
                count_fn = getattr(anchorer, "anchor_count", None)
                if callable(count_fn):
                    now = int(count_fn())
                    if now != int(snap):
                        # The close priced its pool at anchor #snap; the
                        # chain moved, so the on-chain mintPool for this
                        # anchor would differ. Never anchor a stale close.
                        out["anchor_skipped"] = (
                            f"stale snapshot (closed at anchorCount={snap}, "
                            f"chain has {now})")
                        logger.warning(
                            "anchor skipped for epoch %s: %s",
                            fed.epoch_id, out["anchor_skipped"])
                        return None
            return anchorer.prepare_anchor(fed.close_result)
        except Exception as e:
            logger.warning("anchor prepare failed for epoch %s: %s",
                           fed.epoch_id, e)
            out["anchor_skipped"] = f"prepare failed: {e}"
            return None

    def _prepare_pipelined(self, fed: FederatedCloseResult, number: int,
                           out: Dict[str, Any]) -> Optional[Any]:
        """An epoch closed ahead of its parent's anchor: its option chains
        onto the parent as this daemon prepared it, or onto the chain head
        when the parent is anchored. Never for a number already settled,
        or a parent the chain anchored differently."""
        anchorer = self._anchorer
        contract = anchorer._get_contract()
        count = int(contract.functions.anchorCount().call())
        if number <= count:
            out["anchor_skipped"] = f"epoch {number} already settled on chain"
            return None
        prev_root = prev_hash = None
        if number - 1 > count:
            parent = self._fields_by_number.get(number - 1)
            if parent is None:
                out["anchor_skipped"] = f"parent epoch {number - 1} not prepared here"
                return None
            prev_root = bytes(parent.epoch_root)
            prev_hash = bytes(contract.functions.anchorHashOf(*parent.as_args()).call())
        else:
            head = bytes(contract.functions.latestEpochRoot().call()).hex()
            if number > 1 and fed.parent_root and head != fed.parent_root:
                out["anchor_skipped"] = (f"parent root {fed.parent_root[:16]} "
                                         f"is not the chain head {head[:16]}")
                return None
        prepared = anchorer.prepare_anchor(
            fed.close_result, prev_epoch_root=prev_root, prev_anchor_hash=prev_hash)
        self._fields_by_number[number] = prepared.fields
        for n in [n for n in self._fields_by_number if n < count]:
            del self._fields_by_number[n]
        return prepared

    def _publish_anchor_blobs(self, result: AnchorResult) -> None:
        """Every daemon computed the identical anchor blobs; serve them so
        per-agent submitters (and rejoining peers) can fetch them from
        anyone once the anchor lands."""
        for blob, what in ((result.agent_mint_blob, "agent mint"),
                           (result.payload_bytes, "payload")):
            if not blob:
                continue
            try:
                self._blob_resolver.put(blob)
            except Exception as e:
                logger.warning("%s blob put failed: %s", what, e)

    def _anchor_direct(self, prepared: Any) -> AnchorResult:
        result = self._anchorer.submit_prepared(prepared)
        if result.success:
            logger.info(
                "anchor success: epoch=%s tx=%s cid=%s",
                result.epoch_id, result.tx_hash[:16], result.agent_mint_cid[:16],
            )
        else:
            logger.warning("anchor failed: %s", result.error)
        return result

    # ------------------------------------------------------------------
    # Governed anchoring: propose -> vote -> queue -> execute
    # ------------------------------------------------------------------

    def _operator(self) -> Optional[Voter]:
        """The daemon operator key: anchors, proposes, votes with the REP
        the owner delegated to it, and signs household claims."""
        key = self.config.daemon_private_key
        if not key:
            return None
        try:
            from eth_account import Account
            return Voter(Account.from_key(key).address, key)
        except Exception as e:
            logger.warning("daemon operator key unusable: %s", e)
            return None

    def _voters(self) -> List[Voter]:
        """The operator first, then any local agent keys (an agent can
        still hold votes delegated to it)."""
        out: List[Voter] = []
        op = self._operator()
        if op is not None:
            out.append(op)
        try:
            agents = self._agent_resolver()
        except Exception as e:
            logger.warning("agent_chain_resolver failed: %s", e)
            return out
        seen = {v.address.lower() for v in out}
        for a in agents:
            if a.address and a.private_key and a.address.lower() not in seen:
                seen.add(a.address.lower())
                out.append(Voter(a.address, a.private_key))
        return out

    def _step(self, proposal: AnchorProposal, *, may_propose: bool) -> str:
        ga = self.governed_anchorer()
        try:
            state = ga.step(proposal, self._voters(), may_propose=may_propose)
        except Exception as e:
            logger.warning("epoch %s: anchor proposal step failed: %s",
                           proposal.epoch_id, e)
            return proposal.last_state
        logger.info("epoch %s: anchor proposal %s is %s", proposal.epoch_id,
                    hex(proposal.proposal_id)[:12], state)
        return state

    def _step_election(self, election: Election) -> str:
        try:
            state = self._election.step(election, self._voters())
        except Exception as e:
            logger.warning("epoch %s: anchor election step failed: %s",
                           election.epoch_id, e)
            return election.last_state
        logger.info("epoch %s: anchor election (attempt %d) is %s",
                    election.epoch_id, election.attempt, state)
        return state

    def _tick_elections(self) -> None:
        with self._lock:
            elections = list(self._elections.items())
        for epoch_id, election in elections:
            try:
                anchored = self._anchorer.is_anchored(epoch_id)
            except Exception:
                anchored = False
            if not anchored:
                try:
                    count = int(self._anchorer.anchor_count())
                except Exception:
                    count = -1
                if election.epoch_number <= count:
                    # Settled with another option: this one is void.
                    logger.info("epoch %s: settled on chain by another option; "
                                "dropping this daemon's election", epoch_id)
                    with self._lock:
                        self._elections.pop(epoch_id, None)
                    continue
                anchored = self._step_election(election) == "executed"
            done = anchored
            if not anchored and time.monotonic() - election.created > _MAX_PROPOSAL_AGE_S:
                logger.warning("epoch %s: anchor election open too long, "
                               "giving up", epoch_id)
                done = True
            if anchored:
                logger.info("epoch %s anchored by election; submitting mints",
                            epoch_id)
                self._submit_agents(epoch_id)
            if done:
                with self._lock:
                    self._elections.pop(epoch_id, None)

    def tick(self) -> None:
        """Advance every pending anchor proposal and election one step;
        submit agent mints for the ones whose anchor landed."""
        self._tick_elections()
        with self._lock:
            pending = list(self._pending.items())
        for epoch_id, proposal in pending:
            done = False
            try:
                anchored = self._anchorer.is_anchored(epoch_id)
            except Exception:
                anchored = False
            if not anchored:
                # Non-winners propose too once the winner had a tick to.
                age = time.monotonic() - proposal.created
                state = self._step(proposal, may_propose=age >= self.tick_interval)
                anchored = state == "executed"
                if state in _TERMINAL or age > _MAX_PROPOSAL_AGE_S:
                    logger.warning("epoch %s: anchor proposal %s, giving up",
                                   epoch_id, state)
                    done = True
            if anchored:
                logger.info("epoch %s anchored via the DAO; submitting mints",
                            epoch_id)
                self._submit_agents(epoch_id)
                done = True
            if done:
                with self._lock:
                    self._pending.pop(epoch_id, None)

    def _ensure_ticker(self) -> None:
        with self._lock:
            if self._ticker is not None and self._ticker.is_alive():
                return
            self._stop.clear()
            self._ticker = threading.Thread(
                target=self._tick_loop, name="anchor-proposals", daemon=True)
            self._ticker.start()

    def _tick_loop(self) -> None:
        while not self._stop.wait(self.tick_interval):
            try:
                self.tick()
            except Exception:
                logger.warning("anchor proposal tick failed", exc_info=True)
            with self._lock:
                if not self._pending and not self._elections:
                    self._ticker = None
                    return

    def stop(self) -> None:
        self._stop.set()

    @property
    def pending_proposals(self) -> Dict[str, str]:
        with self._lock:
            out = {e: p.last_state for e, p in self._pending.items()}
            out.update({e: el.last_state for e, el in self._elections.items()})
            return out

    def _claimants(self) -> List[str]:
        """The households this daemon claims for: the configured owner
        wallet, each local agent's on-chain owner, and each agent itself
        (an agent bound after the close's owner snapshot is still keyed
        by its own address). A zero share costs one read, no gas."""
        out: List[str] = []
        if self.config.owner_wallet:
            out.append(self.config.owner_wallet)
        try:
            agents = self._agent_resolver()
        except Exception as e:
            logger.warning("agent_chain_resolver failed: %s", e)
            agents = []
        contract = None
        if self._anchorer is not None:
            try:
                contract = self._anchorer._get_contract()
            except Exception:
                contract = None
        for ident in agents:
            if not ident.address:
                continue
            if contract is not None:
                try:
                    owner = contract.functions.getAgentOwner(
                        self._anchorer._w3.to_checksum_address(ident.address)).call()
                    if owner and int(owner, 16):
                        out.append(owner)
                except Exception:
                    pass
            out.append(ident.address)
        seen: set = set()
        unique = []
        for a in out:
            if a.lower() not in seen:
                seen.add(a.lower())
                unique.append(a)
        return unique

    def _submit_agents(self, epoch_id: str) -> List[Dict[str, Any]]:
        """Claim the anchored mint for each household this daemon serves,
        signed by the operator key."""
        op = self._operator()
        if op is None:
            logger.warning("epoch %s: no daemon operator key; household "
                           "claims skipped", epoch_id)
            return []
        results: List[Dict[str, Any]] = []
        for claimant in self._claimants():
            submitter = self._get_submitter(claimant, op)
            try:
                r = submitter.submit_for_epoch(epoch_id)
            except Exception as e:
                logger.error(
                    "submit_for_epoch raised for %s: %s",
                    claimant, e, exc_info=True,
                )
                continue
            results.append({
                "claimant": claimant,
                "success": r.success,
                "tx_hash": r.tx_hash,
                "raw_mint": r.raw_mint,
                "error": r.error,
            })
        return results

    def _submitter_chain(self) -> Dict[str, Any]:
        if self._submitter_web3 is None:
            return {}
        from .contract_artifacts import load_abi
        return {"web3": self._submitter_web3, "substrate_abi": load_abi("Substrate")}

    def _get_submitter(
        self, claimant: str, operator: Voter,
    ) -> AuthoritativeChainSubmitter:
        key = claimant.lower()
        with self._lock:
            cached = self._submitter_cache.get(key)
            if cached is not None:
                return cached
            sub = AuthoritativeChainSubmitter(
                claimant_id=key,
                claimant_address=claimant,
                resolver=self._blob_resolver,
                config=AuthoritativeSubmitterConfig(
                    substrate_address=self.config.substrate_address,
                    rpc_url=self.config.rpc_url,
                    chain_id=self.config.chain_id,
                    private_key=operator.private_key,
                ),
                sender_address=operator.address,
                **self._submitter_chain(),
            )
            self._submitter_cache[key] = sub
            return sub
