"""Authoritative chain submission for one household claimant.

Phase 5.6 of native world-model integration. Replaces Phase 5.1's
``ChainSubmitter`` semantics: we no longer submit the local
projection mint. Instead:

  1. Read the on-chain anchor for the epoch we want to claim from.
  2. Fetch the agent_mint blob from a ``BlobResolver`` (using the
     CID stored in the anchor).
  3. Verify the blob's content hash matches the anchor's CID —
     otherwise the resolver lied or the blob was tampered with.
  4. Decode the claimant's mint share from the blob.
  5. Scale the float mint to ``int`` via ``mint_scale`` (default
     1e6 — see "Two precision boundaries" below).
  6. Call ``Substrate.claimFor(claimant, amount, epochIdHash, proof)``.
     The claimant is the household's owner wallet (Decision 2026-10-27);
     the transaction is signed by whoever submits (the daemon operator
     key), since the leaf already fixes where the ATN goes. The contract
     is per-(claimant, epochId) idempotent, so retries / restarts are
     safe.

Two precision boundaries
------------------------

The system has two intentional float-to-int truncation steps:

  - **Authoritative encoding boundary** (Phase 5.5): the ``agent_mint``
    blob serializes floats as ``f"{x:.10f}"`` — 10 decimal places —
    to guarantee byte-identical CIDs across daemons. This is the
    *consensus precision*.

  - **Chain submission boundary** (Phase 5.6, this module): the float
    is multiplied by ``mint_scale`` (default 1_000_000) and truncated
    to an integer for the ``claimFor(..., uint256 amount, ...)`` call.
    This is the *storage cost precision* — finer scales cost more
    storage; coarser scales drop sub-token mint amounts to zero.

The 4-orders-of-magnitude difference is intentional. Don't try to
collapse them into one boundary.

Phase 5.6 vs 5.1 idempotency
----------------------------

5.1 had agent-side idempotency: the submitter remembered which
epoch_ids it had already submitted, and skipped duplicates. That
was state-fragile (a process restart re-submitted).

5.6 moves idempotency to the contract: ``Substrate.sol`` reverts
``AlreadySubmittedForEpoch`` on a duplicate (claimant, epochIdHash) call.
The submitter still tracks its own seen-set as a fast-path optimization
and to avoid wasting gas on a revert, but the contract is the source
of truth. Restarts won't double-submit because the contract refuses
duplicates regardless of what the submitter remembers.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .authoritative_encoding import (
    cid_for_blob,
    decode_agent_mint_blob,
)
from .blob_resolver import BlobIntegrityError, BlobResolver
from .mint_merkle import mint_merkle_proof, scale_mint, verify_mint_proof


logger = logging.getLogger(__name__)


_DEFAULT_MINT_SCALE = 1_000_000


@dataclass
class AuthoritativeSubmitterConfig:
    """Submitter config. ``private_key`` signs the claim transaction (the
    daemon operator key); it need not belong to the claimant."""
    substrate_address: str
    rpc_url: str = "http://localhost:8545"
    chain_id: int = 31337
    private_key: str = ""
    mint_scale: int = _DEFAULT_MINT_SCALE
    # Whether to silently skip a no-op submission when the claimant's
    # mint share is zero. Saves gas on the common case.
    skip_zero: bool = True


@dataclass
class AuthoritativeSubmissionResult:
    success: bool
    tx_hash: str = ""
    epoch_id: str = ""
    epoch_id_hash_hex: str = ""
    contribution_scaled: int = 0
    raw_mint: float = 0.0
    cid_verified: bool = False
    cid: str = ""
    error: str = ""


def epoch_id_hash(epoch_id: str) -> bytes:
    """keccak256 of the epoch_id string. Matches Substrate.sol's
    ``keccak256(bytes(epochId))`` so on-chain hash lookups line up."""
    # web3.py / pycryptodome both expose keccak; we use eth_utils
    # because it's already a transitive dep.
    from eth_utils import keccak
    return keccak(text=epoch_id)


class AuthoritativeChainSubmitter:
    """Per-claimant helper: read anchor → fetch blob → claim on chain.

    ``claimant_id`` is the key in the anchored mint map (lowercase owner
    wallet) and ``claimant_address`` the address the leaf names.
    ``sender_address`` signs and pays gas; it defaults to the claimant.

    Two construction modes (mirrors Phase 5.1):

      - **Production**: pass ``config`` with rpc_url + private_key. The
        submitter dials HTTP and loads ABI from artifacts.
      - **Tests**: pass ``web3`` + ``substrate_abi`` directly so tests
        run against an in-process EVM without HTTP.
    """

    def __init__(
        self,
        claimant_id: str,
        claimant_address: str,
        resolver: BlobResolver,
        config: AuthoritativeSubmitterConfig,
        *,
        sender_address: Optional[str] = None,
        web3: Optional[Any] = None,
        substrate_abi: Optional[list] = None,
    ):
        self.claimant_id = claimant_id
        self.claimant_address = claimant_address
        self.sender_address = sender_address or claimant_address
        self.resolver = resolver
        self.config = config
        self._lock = threading.RLock()
        self._submitted_epochs: set[str] = set()
        self._w3 = web3
        self._abi = substrate_abi
        self._contract = None

    def _get_contract(self):
        if self._contract is not None:
            return self._contract
        if self._w3 is not None and self._abi is not None:
            self._contract = self._w3.eth.contract(
                address=self._w3.to_checksum_address(self.config.substrate_address),
                abi=self._abi,
            )
            return self._contract

        # Production path
        from .blockchain import BlockchainInterface

        bc = BlockchainInterface(
            rpc_url=self.config.rpc_url,
            chain_id=self.config.chain_id,
            private_key=self.config.private_key,
        )
        from .contract_artifacts import load_abi
        self._abi = load_abi("Substrate")
        self._w3 = bc.web3
        self._contract = bc.web3.eth.contract(
            address=bc.web3.to_checksum_address(self.config.substrate_address),
            abi=self._abi,
        )
        return self._contract

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit_for_epoch(self, epoch_id: str) -> AuthoritativeSubmissionResult:
        """Claim this household's authoritative mint for an anchored epoch.

        Looks up the anchor on chain, fetches the blob, decodes the
        claimant's share, calls ``claimFor``.
        """
        with self._lock:
            if epoch_id in self._submitted_epochs:
                # Fast-path: already submitted in this process. Return
                # success no-op. Source of truth is still the contract.
                return AuthoritativeSubmissionResult(
                    success=True,
                    epoch_id=epoch_id,
                    contribution_scaled=0,
                )

            contract = self._get_contract()

            # 1. Read the anchor on chain.
            try:
                anchor = contract.functions.getAnchorByEpochId(epoch_id).call()
            except Exception as e:
                return AuthoritativeSubmissionResult(
                    success=False,
                    epoch_id=epoch_id,
                    error=f"anchor lookup failed: {e}",
                )
            # Anchor tuple layout matches the Solidity struct order:
            # (epochId, epochRoot, prevEpochRoot, prevAnchorHash,
            #  agentMintCid, payloadHash, submitter, blockNumber,
            #  timestamp, agentMintRoot)
            cid = anchor[4]
            if not cid:
                return AuthoritativeSubmissionResult(
                    success=False,
                    epoch_id=epoch_id,
                    error="anchor has empty agent_mint_cid",
                )

            # 2. Fetch blob.
            blob = self.resolver.get(cid)
            if blob is None:
                return AuthoritativeSubmissionResult(
                    success=False,
                    epoch_id=epoch_id,
                    cid=cid,
                    error=f"blob not found for cid {cid}",
                )

            # 3. Verify CID.
            actual_cid = cid_for_blob(blob)
            if actual_cid != cid:
                raise BlobIntegrityError(
                    f"blob CID mismatch: expected {cid}, got {actual_cid}"
                )

            # 4. Decode the claimant's share (keys are lowercase from
            # schema 6). Scaling MUST route through mint_merkle.scale_mint:
            # the anchor's agentMintRoot was built from the same
            # conversion, and a mismatched integer is an unprovable leaf.
            mint_map = decode_agent_mint_blob(blob)
            raw_mint = float(mint_map.get(
                self.claimant_id, mint_map.get(str(self.claimant_id).lower(), 0.0)))
            scaled = scale_mint(raw_mint, self.config.mint_scale)

            # 5. Skip zero submissions if configured.
            if scaled == 0 and self.config.skip_zero:
                self._submitted_epochs.add(epoch_id)
                return AuthoritativeSubmissionResult(
                    success=True,
                    epoch_id=epoch_id,
                    cid=cid,
                    cid_verified=True,
                    contribution_scaled=0,
                    raw_mint=raw_mint,
                )

            # 6. Build the merkle proof from the full mint map and submit
            # on chain. The leaf commits (claimant, amount), money only.
            proof = mint_merkle_proof(
                mint_map, self.claimant_address, self.config.mint_scale,
            )
            if proof is None:
                return AuthoritativeSubmissionResult(
                    success=False,
                    epoch_id=epoch_id,
                    cid=cid,
                    cid_verified=True,
                    raw_mint=raw_mint,
                    contribution_scaled=scaled,
                    error=(
                        f"no provable mint leaf for {self.claimant_address} "
                        "(claimant id/address mismatch in the mint map?)"
                    ),
                )
            eid_hash = epoch_id_hash(epoch_id)

            # 6b. GUARD: pre-flight-verify the amount we are about to send
            # against the anchor's ON-CHAIN root. Root, proof, and tx all
            # derive from the same blob's mint map, so this only fails if
            # the anchorer committed a root built from a different map than
            # the blob it published (or the blob decode drifted) — fail
            # loud here instead of burning gas on MintProofInvalid.
            onchain_root = bytes(anchor[9])
            if not verify_mint_proof(
                onchain_root, self.claimant_address, scaled, proof,
            ):
                return AuthoritativeSubmissionResult(
                    success=False,
                    epoch_id=epoch_id,
                    cid=cid,
                    cid_verified=True,
                    raw_mint=raw_mint,
                    contribution_scaled=scaled,
                    epoch_id_hash_hex=eid_hash.hex(),
                    error=(
                        "pre-flight mint proof failed against the on-chain "
                        f"agentMintRoot for amount={scaled} — the anchored "
                        "root and the published blob disagree"
                    ),
                )

            # Contract-side dedupe: after a restart our seen-set is empty
            # but the household may already be claimed (by us or by any
            # other submitter). Ask the contract rather than reading a
            # revert, so a real failure (pool exceeded, bad proof) is never
            # mistaken for "already claimed".
            if self._already_claimed(eid_hash):
                self._submitted_epochs.add(epoch_id)
                return AuthoritativeSubmissionResult(
                    success=True,
                    epoch_id=epoch_id,
                    cid=cid,
                    cid_verified=True,
                    contribution_scaled=0,
                    raw_mint=raw_mint,
                    error="already claimed on chain (no-op)",
                )
            try:
                tx_hash = self._send_record(scaled, eid_hash, proof)
            except Exception as e:
                # Lost a race with another submitter for the same household.
                err_str = str(e).lower()
                if "alreadysubmittedforepoch" in err_str or self._already_claimed(eid_hash):
                    self._submitted_epochs.add(epoch_id)
                    return AuthoritativeSubmissionResult(
                        success=True,
                        epoch_id=epoch_id,
                        cid=cid,
                        cid_verified=True,
                        contribution_scaled=0,
                        raw_mint=raw_mint,
                        error=f"already claimed on chain (no-op): {e}",
                    )
                return AuthoritativeSubmissionResult(
                    success=False,
                    epoch_id=epoch_id,
                    cid=cid,
                    cid_verified=True,
                    contribution_scaled=scaled,
                    raw_mint=raw_mint,
                    epoch_id_hash_hex=eid_hash.hex(),
                    error=f"claimFor failed: {e}",
                )

            self._submitted_epochs.add(epoch_id)
            logger.info(
                "Authoritative claim: claimant=%s, epoch=%s, "
                "raw=%.10f, scaled=%d, tx=%s",
                self.claimant_id, epoch_id, raw_mint, scaled, tx_hash,
            )
            return AuthoritativeSubmissionResult(
                success=True,
                tx_hash=tx_hash,
                epoch_id=epoch_id,
                epoch_id_hash_hex=eid_hash.hex(),
                contribution_scaled=scaled,
                raw_mint=raw_mint,
                cid=cid,
                cid_verified=True,
            )

    def _already_claimed(self, eid_hash: bytes) -> bool:
        try:
            return bool(self._get_contract().functions.hasSubmittedForEpoch(
                self._w3.to_checksum_address(self.claimant_address), eid_hash,
            ).call())
        except Exception:
            return False

    def _send_record(
        self, contribution: int, eid_hash: bytes, proof: list,
    ) -> str:
        contract = self._get_contract()
        w3 = self._w3
        if w3 is None:
            raise RuntimeError("submitter has no Web3 instance")

        # claimFor(claimant, amount, epochIdHash, proof). ``contribution``
        # is the ATN mint; the leaf commits (claimant, amount). REP is
        # claimed DAO-side.
        call = contract.functions.claimFor(
            w3.to_checksum_address(self.claimant_address),
            contribution, eid_hash, proof,
        )
        sender = w3.to_checksum_address(self.sender_address)

        # Test path: account is unlocked in eth_tester.
        if not self.config.private_key:
            tx_hash = call.transact({"from": sender, "gas": 1_500_000})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status != 1:
                raise RuntimeError(f"tx reverted: {receipt}")
            return tx_hash.hex()

        # Production path: estimate gas with a margin so we work on
        # networks with higher actual gas costs (e.g. Etherlink shadownet
        # reports ~700K for the claim). A flat hardcode bites in
        # real-world deployments.
        try:
            estimated = call.estimate_gas({"from": sender})
            gas_limit = max(int(estimated * 12 // 10), 1_500_000)
        except Exception:
            gas_limit = 2_000_000
        tx = call.build_transaction({
            "from": sender,
            "nonce": w3.eth.get_transaction_count(sender),
            "gas": gas_limit,
            "gasPrice": w3.eth.gas_price,
            "chainId": self.config.chain_id,
        })
        signed = w3.eth.account.sign_transaction(tx, self.config.private_key)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = w3.eth.send_raw_transaction(raw)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise RuntimeError(f"tx reverted: {receipt}")
        return tx_hash.hex()
