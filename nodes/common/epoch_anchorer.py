"""Agent-side anchor submitter for the substrate-native chain surface.

Runs inside a daemon process but signs from an agent's keypair —
the on-chain submitter address is always an agent's, not a daemon's.

Phase 5.5 of native world-model integration. Updated in 5.6a to
target the consolidated ``Substrate.sol`` contract (which absorbed
the standalone EpochAnchor.sol after the pre-substrate contract
nuke).

After a federated epoch close, one agent (acting as the rotating
canonical-ordering coordinator for that epoch) submits the
authoritative payload to ``Substrate.anchor`` from their own
keypair. Other agents in the federation verify the on-chain record
matches their local computation; mismatches indicate a fork and
trigger a dispute (later phase). Daemons are the wire transport,
not the consensus actor — the on-chain submitter is always an agent
address.

What this module does
---------------------

  - Reads the on-chain ``latestAnchorHash`` and ``latestEpochRoot``
    so the new anchor links into the chain correctly.
  - Encodes the authoritative payload via the canonical encoding spec.
  - Encodes the agent_mint blob, computes its CID.
  - Anchors via ``Substrate.anchor(...)`` (governor only; see governed_anchor).
  - Returns an AnchorResult with the on-chain tx hash, the encoded
    payload bytes (so the caller can serve the blob over P2P), and
    the CID.

What this module does NOT do
----------------------------

  - Serve the off-chain blob — that's blob_store + libp2p in another
    layer.
  - Resolve forks. If the anchor submission reverts because another
    agent already submitted for this epoch, the caller treats that
    as "I'm second; verify their anchor matches mine and accept it,
    else dispute". Phase 5+ wires that.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .authoritative_encoding import (
    cid_for_blob,
    encode_agent_mint_blob,
    encode_authoritative_payload,
    payload_hash,
)
from .mint_merkle import mint_merkle_root
from .tx_nonce import next_nonce


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorFields:
    """The eight arguments of ``Substrate.anchor``."""
    epoch_id: str
    epoch_root: bytes
    prev_epoch_root: bytes
    prev_anchor_hash: bytes
    agent_mint_cid: str
    payload_hash: bytes
    agent_mint_root: bytes
    # The epoch's cutoff (Decision 2026-10-26): the one its id carries, or
    # for an id without one (local/library use) the chain head's timestamp.
    cutoff: int = 0

    def as_args(self) -> tuple:
        return (self.epoch_id, self.epoch_root, self.prev_epoch_root,
                self.prev_anchor_hash, self.agent_mint_cid,
                self.payload_hash, self.agent_mint_root, int(self.cutoff))


@dataclass
class EpochAnchorerConfig:
    """Address of the deployed Substrate contract.

    Field name kept as ``epoch_anchor_address`` for callsite stability
    even though the chain surface is now the consolidated
    Substrate.sol (which absorbed EpochAnchor's responsibilities)."""
    epoch_anchor_address: str
    rpc_url: str = "http://localhost:8545"
    chain_id: int = 31337
    private_key: str = ""


@dataclass
class AnchorResult:
    success: bool
    tx_hash: str = ""
    epoch_id: str = ""
    epoch_root_hex: str = ""
    prev_epoch_root_hex: str = ""
    agent_mint_cid: str = ""
    agent_mint_blob: bytes = b""
    payload_bytes: bytes = b""
    payload_hash_hex: str = ""
    agent_mint_root_hex: str = ""
    error: str = ""


@dataclass
class PreparedAnchor:
    """Everything needed to anchor one epoch. ``result`` carries the
    encoded blobs (success=False until anchored)."""
    fields: AnchorFields
    result: AnchorResult


class EpochAnchorer:
    """Anchors federated epoch closes to ``Substrate.sol``.

    Thread-safe: a single anchorer instance can be called from the
    daemon's worker thread or an asyncio bridge.
    """

    def __init__(
        self,
        config: EpochAnchorerConfig,
        *,
        web3: Optional[Any] = None,
        contract_abi: Optional[list] = None,
    ):
        """Two construction modes.

        Production: pass ``config`` with rpc_url + private_key; the
        anchorer dials the RPC and loads the ABI from artifacts.

        Tests: pass ``web3`` + ``contract_abi`` directly so tests can
        run against an in-process EVM (eth_tester) without HTTP.
        """
        self.config = config
        self._lock = threading.RLock()
        self._w3 = web3
        self._abi = contract_abi
        self._contract = None
        self._submitter_address: Optional[str] = None

    def _get_contract(self):
        if self._contract is not None:
            return self._contract

        if self._w3 is not None and self._abi is not None:
            self._contract = self._w3.eth.contract(
                address=self._w3.to_checksum_address(self.config.epoch_anchor_address),
                abi=self._abi,
            )
            return self._contract

        # Production path: build via blockchain.BlockchainInterface.
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
            address=bc.web3.to_checksum_address(self.config.epoch_anchor_address),
            abi=self._abi,
        )
        return self._contract

    def _set_submitter_address(self, addr: str) -> None:
        """Override the submitter address (for tests where the caller
        has multiple accounts and wants to pick one)."""
        with self._lock:
            self._submitter_address = addr

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def latest_anchor_hash(self) -> bytes:
        contract = self._get_contract()
        h = contract.functions.latestAnchorHash().call()
        return bytes(h)

    def latest_epoch_root(self) -> bytes:
        contract = self._get_contract()
        r = contract.functions.latestEpochRoot().call()
        return bytes(r)

    def is_anchored(self, epoch_id: str) -> bool:
        contract = self._get_contract()
        return bool(contract.functions.isAnchored(epoch_id).call())

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def chain_id(self) -> int:
        """The chain id the EIP-712 domain binds to (read from the node,
        cached; falls back to config when the node cannot answer)."""
        cached = getattr(self, "_chain_id_cache", None)
        if cached is not None:
            return cached
        self._get_contract()
        try:
            cid = int(self._w3.eth.chain_id)
        except Exception:
            cid = int(self.config.chain_id)
        self._chain_id_cache = cid
        return cid

    def prepare_anchor(self, close_result: Dict[str, Any], *,
                       prev_epoch_root: Optional[bytes] = None,
                       prev_anchor_hash: Optional[bytes] = None) -> PreparedAnchor:
        """Encode a federated close result into the anchor arguments. Reads
        the chain head (latestEpochRoot / latestAnchorHash), so honest
        daemons reading the same head derive the identical anchor (and so
        the identical DAO proposal). Raises ValueError on a malformed close
        result."""
        with self._lock:
            payload = close_result.get("authoritative_payload")
            if payload is None:
                raise ValueError("close_result has no authoritative_payload")

            epoch_id = str(close_result.get("epoch_id") or "")
            if not epoch_id:
                # Fall back to a deterministic id derived from the root.
                epoch_id = f"e_{payload.get('epoch_root', '')[:16]}"
            epoch_root_hex = str(payload.get("epoch_root", ""))
            if not epoch_root_hex:
                raise ValueError("authoritative_payload missing epoch_root")

            # A pipelined epoch chains onto its parent as this daemon closed
            # it (not anchored yet): the caller passes that parent's root and
            # its content-only anchor hash. Otherwise, the chain head.
            prev_root_bytes = (bytes(prev_epoch_root) if prev_epoch_root is not None
                               else self.latest_epoch_root())
            prev_anchor_bytes = (bytes(prev_anchor_hash) if prev_anchor_hash is not None
                                 else self.latest_anchor_hash())
            prev_root_hex = prev_root_bytes.hex()

            # Encode the authoritative payload canonically.
            try:
                payload_bytes = encode_authoritative_payload(
                    payload,
                    epoch_id=epoch_id,
                    epoch_root_hex=epoch_root_hex,
                    prev_epoch_root_hex=prev_root_hex,
                )
            except Exception as e:
                raise ValueError(f"encoding failed: {e}") from e

            ph = payload_hash(payload_bytes)

            # Encode + CID the off-chain agent_mint blob (money only,
            # Decision 2026-07-10 — no agent_rep map; REP is DAO-claimed).
            agent_mint = close_result.get("agent_mint") or payload.get("agent_mint", {})
            mint_blob = encode_agent_mint_blob(agent_mint)
            cid = cid_for_blob(mint_blob)

            # Merkle root over the mint map: each household's share is
            # proven against this when claimed (claimFor), a 2-field
            # (claimant, amount) leaf.
            mint_root = mint_merkle_root(agent_mint)

            from .voice_state import epoch_id_cutoff
            cutoff = epoch_id_cutoff(epoch_id)
            if cutoff is None:
                contract = self._get_contract()
                cutoff = max(
                    int(self._w3.eth.get_block("latest")["timestamp"]),
                    int(contract.functions.latestCutoff().call()) + 1)

            fields = AnchorFields(
                epoch_id=epoch_id,
                epoch_root=bytes.fromhex(epoch_root_hex),
                prev_epoch_root=prev_root_bytes,
                prev_anchor_hash=prev_anchor_bytes,
                agent_mint_cid=cid,
                payload_hash=ph,
                agent_mint_root=mint_root,
                cutoff=cutoff,
            )
            result = AnchorResult(
                success=False,
                epoch_id=epoch_id,
                epoch_root_hex=epoch_root_hex,
                prev_epoch_root_hex=prev_root_hex,
                agent_mint_cid=cid,
                agent_mint_blob=mint_blob,
                payload_bytes=payload_bytes,
                payload_hash_hex=ph.hex(),
                agent_mint_root_hex=mint_root.hex(),
            )
            return PreparedAnchor(fields=fields, result=result)

    def anchor_count(self) -> int:
        return int(self._get_contract().functions.anchorCount().call())

    def submit_prepared(
        self,
        prepared: PreparedAnchor,
        *,
        from_address: Optional[str] = None,
    ) -> AnchorResult:
        """Anchor directly. Only Substrate's governor may: the DAO timelock
        in production (see governed_anchor), an EOA on dev networks."""
        with self._lock:
            r = prepared.result
            try:
                r.tx_hash = self._submit(
                    *prepared.fields.as_args(), from_address=from_address)
                r.success = True
                r.error = ""
            except Exception as e:
                r.success = False
                r.error = f"anchor reverted: {e}"
            return r

    def anchor_close_result(
        self,
        close_result: Dict[str, Any],
        *,
        from_address: Optional[str] = None,
        signatures: Optional[List[bytes]] = None,
        signer_keys: Optional[List[str]] = None,
    ) -> AnchorResult:
        """Prepare + anchor directly in one call (sender must be the
        governor). ``signatures`` / ``signer_keys`` are accepted and ignored:
        the REP-signature quorum they fed is gone (Decision 2026-10-16)."""
        with self._lock:
            try:
                prepared = self.prepare_anchor(close_result)
            except ValueError as e:
                return AnchorResult(success=False, error=str(e))
            return self.submit_prepared(prepared, from_address=from_address)

    def _submit(
        self,
        epoch_id: str,
        epoch_root: bytes,
        prev_epoch_root: bytes,
        prev_anchor_hash: bytes,
        agent_mint_cid: str,
        payload_hash_bytes: bytes,
        agent_mint_root: bytes,
        cutoff: int,
        *,
        from_address: Optional[str] = None,
    ) -> str:
        contract = self._get_contract()
        w3 = self._w3
        if w3 is None:
            raise RuntimeError("EpochAnchorer has no Web3 instance")

        # Resolve sender: explicit override > previously-set address >
        # derive from private key (production) > eth_accounts[0]
        # (eth_tester only, where the RPC actually exposes a managed
        # wallet).
        sender = from_address or self._submitter_address
        if sender is None and self.config.private_key:
            from eth_account import Account
            pk = self.config.private_key
            if not pk.startswith("0x"):
                pk = "0x" + pk
            sender = Account.from_key(pk).address
        if sender is None:
            # eth_tester fallback. eth_accounts is disabled on most
            # public RPCs (e.g. Etherlink shadownet), so we only
            # call it when nothing else worked.
            try:
                accts = w3.eth.accounts
            except Exception:
                accts = []
            sender = accts[0] if accts else None
        if sender is None:
            raise RuntimeError("no submitter address available")

        # Test path: account is unlocked in eth_tester, sign via transact.
        if not self.config.private_key:
            tx_hash = contract.functions.anchor(
                epoch_id,
                epoch_root,
                prev_epoch_root,
                prev_anchor_hash,
                agent_mint_cid,
                payload_hash_bytes,
                agent_mint_root,
                int(cutoff),
            ).transact({"from": sender})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status != 1:
                raise RuntimeError(f"tx reverted: {receipt}")
            logger.info(
                "Anchor submitted: epoch=%s, tx=%s", epoch_id, tx_hash.hex(),
            )
            return tx_hash.hex()

        # Production path: sign with private key. Estimate gas with a
        # margin so we work on networks with higher actual costs (e.g.
        # Etherlink shadownet reports ~1.2M for an anchor). A flat
        # hardcode bites in real-world deployments.
        try:
            estimated = contract.functions.anchor(
                epoch_id,
                epoch_root,
                prev_epoch_root,
                prev_anchor_hash,
                agent_mint_cid,
                payload_hash_bytes,
                agent_mint_root,
                int(cutoff),
            ).estimate_gas({"from": sender})
            gas_limit = max(int(estimated * 12 // 10), 1_500_000)
        except Exception:
            gas_limit = 2_500_000
        tx = contract.functions.anchor(
            epoch_id,
            epoch_root,
            prev_epoch_root,
            prev_anchor_hash,
            agent_mint_cid,
            payload_hash_bytes,
            agent_mint_root,
            int(cutoff),
        ).build_transaction({
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
        logger.info("Anchor submitted: epoch=%s, tx=%s", epoch_id, tx_hash.hex())
        return tx_hash.hex()
