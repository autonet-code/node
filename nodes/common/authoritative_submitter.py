"""Authoritative chain claims for one household earner.

For an anchored epoch:

  1. Read the anchor on chain (``getAnchorByEpochId``).
  2. Fetch the claims blob the anchor's ``claimCid`` names from a
     ``BlobResolver`` and check its sha256 against the cid: otherwise the
     resolver lied or the blob was tampered with.
  3. Read the earner's leaf (its amount of each token's pool) and build its
     proof from the whole blob (claims and concluded tools share the tree),
     then check the proof against the anchor's ``claimRoot`` before
     spending gas.
  4. Call ``Substrate.claim(earner, epochIdHash, tokens, amounts, proof)``.
     The earner is the household's owner wallet (Decision 2026-10-27); the
     transaction is signed by whoever submits (the daemon operator key),
     since the leaf already fixes the earner. The contract is idempotent
     per (earner, epoch): ``claimedEpoch`` is checked first, so retries and
     restarts never double-claim.
  5. ``withdraw(earner, token)`` for each token the earner is owed
     (``owed(earner, token) > 0``). A token whose withdrawal fails (paused,
     blocked) is logged and retried at the next pass; it never blocks the
     others.

Amounts are integers in each token's base units from end to end: the blob
carries them as decimal strings and the leaf commits them as they are.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .authoritative_encoding import (
    cid_for_blob,
    decode_claims_blob,
    decode_concluded,
)
from .blob_resolver import BlobIntegrityError, BlobResolver
from .mint_merkle import claim_entry, claim_merkle_proof, verify_claim_proof
from .tx_nonce import next_nonce


logger = logging.getLogger(__name__)

# Substrate.Anchor tuple indices used here.
_ANCHOR_CLAIM_CID = 4
_ANCHOR_CLAIM_ROOT = 9


@dataclass
class AuthoritativeSubmitterConfig:
    """Submitter config. ``private_key`` signs the claim and withdrawal
    transactions (the daemon operator key); it need not belong to the
    earner."""
    substrate_address: str
    rpc_url: str = "http://localhost:8545"
    chain_id: int = 31337
    private_key: str = ""
    # Skip an epoch in which the earner has no leaf (the common case).
    skip_zero: bool = True


@dataclass
class AuthoritativeSubmissionResult:
    success: bool
    tx_hash: str = ""
    epoch_id: str = ""
    epoch_id_hash_hex: str = ""
    # The leaf claimed: lowercase token -> amount (base units), and its
    # informational value (6-decimal value units, from the blob).
    tokens: Dict[str, int] = field(default_factory=dict)
    value: int = 0
    cid_verified: bool = False
    cid: str = ""
    # Withdrawals made in this pass: token -> tx hash; and the tokens whose
    # withdrawal failed (retried at the next pass): token -> error.
    withdrawn: Dict[str, str] = field(default_factory=dict)
    withdraw_errors: Dict[str, str] = field(default_factory=dict)
    error: str = ""


def epoch_id_hash(epoch_id: str) -> bytes:
    """keccak256 of the epoch_id string. Matches Substrate.sol's
    ``keccak256(bytes(epochId))`` so on-chain hash lookups line up."""
    from eth_utils import keccak
    return keccak(text=epoch_id)


class AuthoritativeChainSubmitter:
    """Per-earner helper: read anchor -> fetch blob -> claim -> withdraw.

    ``claimant_id`` is the key in the anchored claims (lowercase owner
    wallet) and ``claimant_address`` the address the leaf names.
    ``sender_address`` signs and pays gas; it defaults to the earner.

    Two construction modes:

      - **Production**: pass ``config`` with rpc_url + private_key. The
        submitter dials HTTP and loads the ABI from artifacts.
      - **Tests**: pass ``web3`` + ``substrate_abi`` directly so tests run
        against an in-process EVM without HTTP.
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
        # Tokens the earner was owed whose withdrawal has not gone through
        # yet: retried at every pass.
        self._pending_withdrawals: set[str] = set()
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
        """Claim this household's leaf of an anchored epoch, then withdraw
        every token it is owed (this epoch's and any earlier failures)."""
        with self._lock:
            r = self._claim(epoch_id)
            if r.success:
                self._withdraw_owed(set(r.tokens), r)
            return r

    def _claim(self, epoch_id: str) -> AuthoritativeSubmissionResult:
        if epoch_id in self._submitted_epochs:
            # Fast path: already claimed in this process (the contract is
            # still the source of truth).
            return AuthoritativeSubmissionResult(success=True, epoch_id=epoch_id)

        contract = self._get_contract()
        try:
            anchor = contract.functions.getAnchorByEpochId(epoch_id).call()
        except Exception as e:
            return AuthoritativeSubmissionResult(
                success=False, epoch_id=epoch_id, error=f"anchor lookup failed: {e}")
        cid = anchor[_ANCHOR_CLAIM_CID]
        if not cid:
            return AuthoritativeSubmissionResult(
                success=False, epoch_id=epoch_id, error="anchor has an empty claimCid")

        blob = self.resolver.get(cid)
        if blob is None:
            return AuthoritativeSubmissionResult(
                success=False, epoch_id=epoch_id, cid=cid,
                error=f"blob not found for cid {cid}")
        actual_cid = cid_for_blob(blob)
        if actual_cid != cid:
            raise BlobIntegrityError(
                f"blob CID mismatch: expected {cid}, got {actual_cid}")

        claims = decode_claims_blob(blob)
        concluded = decode_concluded(blob)
        entry = claims.get(str(self.claimant_id).lower()) or claims.get(self.claimant_id) or {}
        value = int(entry.get("value") or 0)
        leaf = claim_entry(claims, self.claimant_address)
        if leaf is None:
            if self.config.skip_zero:
                self._submitted_epochs.add(epoch_id)
                return AuthoritativeSubmissionResult(
                    success=True, epoch_id=epoch_id, cid=cid, cid_verified=True)
            return AuthoritativeSubmissionResult(
                success=False, epoch_id=epoch_id, cid=cid, cid_verified=True,
                error=f"no claim leaf for {self.claimant_address}")
        _earner, tokens, amounts = leaf
        tokens_map = {t.lower(): a for t, a in zip(tokens, amounts)}
        base = dict(epoch_id=epoch_id, cid=cid, cid_verified=True,
                    tokens=tokens_map, value=value)

        proof = claim_merkle_proof(claims, self.claimant_address, concluded=concluded) or []
        eid_hash = epoch_id_hash(epoch_id)
        # Pre-flight the leaf against the anchor's ON-CHAIN root. Root,
        # proof and tx all derive from the same blob, so this fails only
        # if the anchor committed a root over different claims than the
        # blob it published: fail loud instead of burning gas.
        if not verify_claim_proof(bytes(anchor[_ANCHOR_CLAIM_ROOT]),
                                  self.claimant_address, tokens, amounts, proof):
            return AuthoritativeSubmissionResult(
                success=False, epoch_id_hash_hex=eid_hash.hex(),
                error=("pre-flight claim proof failed against the on-chain "
                       "claimRoot: the anchored root and the published blob "
                       "disagree"), **base)

        if self._already_claimed(eid_hash):
            self._submitted_epochs.add(epoch_id)
            return AuthoritativeSubmissionResult(
                success=True, epoch_id_hash_hex=eid_hash.hex(),
                error="already claimed on chain (no-op)", **base)
        try:
            tx_hash = self._send(contract.functions.claim(
                self._w3.to_checksum_address(self.claimant_address), eid_hash,
                list(tokens), [int(a) for a in amounts], proof))
        except Exception as e:
            if self._already_claimed(eid_hash):
                # Lost a race with another submitter for the same household.
                self._submitted_epochs.add(epoch_id)
                return AuthoritativeSubmissionResult(
                    success=True, epoch_id_hash_hex=eid_hash.hex(),
                    error=f"already claimed on chain (no-op): {e}", **base)
            return AuthoritativeSubmissionResult(
                success=False, epoch_id_hash_hex=eid_hash.hex(),
                error=f"claim failed: {e}", **base)

        self._submitted_epochs.add(epoch_id)
        logger.info("claim: earner=%s, epoch=%s, tokens=%s, value=%d, tx=%s",
                    self.claimant_id, epoch_id, tokens_map, value, tx_hash)
        return AuthoritativeSubmissionResult(
            success=True, tx_hash=tx_hash, epoch_id_hash_hex=eid_hash.hex(), **base)

    def withdraw_owed(self, tokens: Optional[List[str]] = None) -> AuthoritativeSubmissionResult:
        """Withdraw what the earner is owed in ``tokens`` and in every token
        whose earlier withdrawal failed."""
        with self._lock:
            r = AuthoritativeSubmissionResult(success=True)
            self._withdraw_owed({str(t).lower() for t in (tokens or [])}, r)
            return r

    def _withdraw_owed(self, tokens: set, r: AuthoritativeSubmissionResult) -> None:
        contract = self._get_contract()
        earner = self._w3.to_checksum_address(self.claimant_address)
        for token in sorted(set(tokens) | self._pending_withdrawals):
            t = self._w3.to_checksum_address(token)
            try:
                owed = int(contract.functions.owed(earner, t).call())
                if owed <= 0:
                    self._pending_withdrawals.discard(token)
                    continue
                r.withdrawn[token] = self._send(contract.functions.withdraw(earner, t))
                self._pending_withdrawals.discard(token)
                logger.info("withdrawn: earner=%s, token=%s, amount=%d",
                            self.claimant_id, token, owed)
            except Exception as e:  # noqa: BLE001 (one token never blocks the rest)
                self._pending_withdrawals.add(token)
                r.withdraw_errors[token] = str(e)
                logger.warning("withdrawal of %s for %s failed (retried next pass): %s",
                               token, self.claimant_id, e)

    def _already_claimed(self, eid_hash: bytes) -> bool:
        try:
            return bool(self._get_contract().functions.claimedEpoch(
                self._w3.to_checksum_address(self.claimant_address), eid_hash,
            ).call())
        except Exception:
            return False

    def _send(self, call) -> str:
        w3 = self._w3
        if w3 is None:
            raise RuntimeError("submitter has no Web3 instance")
        sender = w3.to_checksum_address(self.sender_address)

        # Test path: account is unlocked in eth_tester.
        if not self.config.private_key:
            tx_hash = call.transact({"from": sender, "gas": 1_500_000})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status != 1:
                raise RuntimeError(f"tx reverted: {receipt}")
            return tx_hash.hex()

        # Production path: estimate gas with a margin (public networks
        # report more than a local node).
        try:
            estimated = call.estimate_gas({"from": sender})
            gas_limit = max(int(estimated * 12 // 10), 300_000)
        except Exception:
            gas_limit = 2_000_000
        tx = call.build_transaction({
            "from": sender,
            "nonce": next_nonce(w3, sender),
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
