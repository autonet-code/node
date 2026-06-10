"""Chain submission driver — anchor + per-agent mint submission.

Triggered by ``FederatedCloseDriver`` results (Phase 10.3a). Two
responsibilities:

  - **Anchor**: when this daemon is the deterministic-random winner
    for an epoch, encode the close result and submit the anchor to
    Substrate.sol. Only one daemon submits per epoch; the contract's
    ``isAnchored(epoch_id)`` rejects duplicates as defense in depth.

  - **Per-agent mint**: for every locally-registered agent (each one
    holds its own private key + 0x address via the agent registry),
    construct an ``AuthoritativeChainSubmitter`` and call
    ``submit_for_epoch(epoch_id)``. The submitter reads the anchor
    on chain, fetches the blob, decodes the agent's share, and
    submits ``recordTrainingForEpoch``. Per-agent submission is
    inherently parallel — every agent does it independently.

The anchor blob (off-chain) is written to a ``BlobResolver`` that
the per-agent submitters read from. For single-process testing an
``InMemoryBlobResolver`` is fine; cross-daemon needs a libp2p-backed
resolver (Phase 10.3d).
"""

from __future__ import annotations

import logging
import threading
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


logger = logging.getLogger(__name__)


@dataclass
class ChainSubmissionConfig:
    """Per-daemon chain config. Mirrors BlockchainConfig but renamed
    so the submitter doesn't import the broader autonet config."""
    substrate_address: str
    rpc_url: str = "http://localhost:8545"
    chain_id: int = 31337
    daemon_private_key: str = ""


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
    ):
        self.config = config
        self._agent_resolver = agent_chain_resolver
        # Use `is None` rather than `or` because BlobResolver impls
        # define ``__len__`` and an empty resolver evaluates falsy.
        self._blob_resolver = (
            blob_resolver if blob_resolver is not None else InMemoryBlobResolver()
        )
        self._lock = threading.RLock()
        # Cache: agent_address -> AuthoritativeChainSubmitter.
        self._submitter_cache: Dict[str, AuthoritativeChainSubmitter] = {}
        # Anchor side. Configurable so tests can pass an in-process
        # anchorer with eth_tester.
        self._anchorer = anchorer or self._build_anchorer()

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

        if fed.is_winner:
            anchor_result = self._anchor(fed)
            out["anchored"] = anchor_result.success
            out["anchor_tx"] = anchor_result.tx_hash
            if not anchor_result.success:
                out["anchor_error"] = anchor_result.error

        # Per-agent submissions. Run regardless of is_winner — every
        # daemon submits each of its agents' mints; the anchor must
        # already be on chain (winner posted it just above on the
        # winner daemon, or another daemon raced ahead).
        agent_results = self._submit_agents(fed.epoch_id)
        out["agent_submissions"] = agent_results
        return out

    def _anchor(self, fed: FederatedCloseResult) -> AnchorResult:
        try:
            result = self._anchorer.anchor_close_result(fed.close_result)
        except Exception as e:
            logger.error("anchor failed: %s", e, exc_info=True)
            return AnchorResult(success=False, error=str(e))
        if result.success and result.agent_mint_blob:
            # Make the blob fetchable by per-agent submitters (and,
            # eventually, by peers via the libp2p resolver).
            try:
                self._blob_resolver.put(result.agent_mint_blob)
            except Exception as e:
                logger.warning("blob_resolver.put failed: %s", e)
        if result.success and result.payload_bytes:
            # Publish the encoded authoritative payload too: its cid
            # equals the on-chain payloadHash, which is how rejoining
            # daemons discover world_cid during anchored catch-up.
            try:
                self._blob_resolver.put(result.payload_bytes)
            except Exception as e:
                logger.warning("payload blob put failed: %s", e)
        if result.success:
            logger.info(
                "anchor success: epoch=%s tx=%s cid=%s",
                result.epoch_id, result.tx_hash[:16], result.agent_mint_cid[:16],
            )
        else:
            logger.warning("anchor failed: %s", result.error)
        return result

    def _submit_agents(self, epoch_id: str) -> List[Dict[str, Any]]:
        """Run AuthoritativeChainSubmitter.submit_for_epoch for each
        registered agent on this daemon."""
        try:
            agents = self._agent_resolver()
        except Exception as e:
            logger.warning("agent_chain_resolver failed: %s", e)
            return []

        results: List[Dict[str, Any]] = []
        for ident in agents:
            if not ident.address or not ident.private_key:
                continue
            submitter = self._get_submitter(ident)
            try:
                r = submitter.submit_for_epoch(epoch_id)
            except Exception as e:
                logger.error(
                    "submit_for_epoch raised for %s: %s",
                    ident.address, e, exc_info=True,
                )
                continue
            results.append({
                "agent_address": ident.address,
                "success": r.success,
                "tx_hash": r.tx_hash,
                "raw_mint": r.raw_mint,
                "error": r.error,
            })
        return results

    def _get_submitter(
        self, ident: AgentChainIdentity,
    ) -> AuthoritativeChainSubmitter:
        with self._lock:
            cached = self._submitter_cache.get(ident.address)
            if cached is not None:
                return cached
            sub = AuthoritativeChainSubmitter(
                agent_id=ident.address,
                agent_address=ident.address,
                resolver=self._blob_resolver,
                config=AuthoritativeSubmitterConfig(
                    substrate_address=self.config.substrate_address,
                    rpc_url=self.config.rpc_url,
                    chain_id=self.config.chain_id,
                    private_key=ident.private_key,
                ),
            )
            self._submitter_cache[ident.address] = sub
            return sub
