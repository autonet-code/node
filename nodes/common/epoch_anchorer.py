"""Agent-side anchor submitter for the substrate-native chain surface.

Runs inside a daemon process but signs from an agent's keypair —
the on-chain submitter address is always an agent's, not a daemon's.

Phase 5.5 of native world-model integration. Updated in 5.6a to
target the consolidated ``Substrate.sol`` contract (which absorbed
the standalone EpochAnchor.sol after the pre-substrate contract
nuke).

After a federated epoch close, one agent (acting as the rotating
canonical-ordering coordinator for that epoch) submits the
authoritative payload to ``Substrate.submitAnchor`` from their own
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
  - Submits ``submitAnchor(...)`` to the EpochAnchor contract.
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
from typing import Any, Dict, Optional

from .authoritative_encoding import (
    cid_for_blob,
    encode_agent_mint_blob,
    encode_authoritative_payload,
    payload_hash,
)
from .mint_merkle import mint_merkle_root


logger = logging.getLogger(__name__)


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
        import json as _json
        from pathlib import Path

        bc = BlockchainInterface(
            rpc_url=self.config.rpc_url,
            chain_id=self.config.chain_id,
            private_key=self.config.private_key,
        )
        artifact_path = Path("C:/code/autonet/artifacts/contracts/core/Substrate.sol/Substrate.json")
        with artifact_path.open("r", encoding="utf-8") as fh:
            artifact = _json.load(fh)
        self._abi = artifact["abi"]
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

    def anchor_close_result(
        self,
        close_result: Dict[str, Any],
        *,
        from_address: Optional[str] = None,
    ) -> AnchorResult:
        """Submit a federated close result to the chain.

        ``close_result`` is the dict returned by
        ``federated_epoch_close``. We extract:
          - ``authoritative_payload``: the chain-anchorable subset
          - ``epoch_root``: hex string
          - ``agent_mint``: dict for the off-chain blob

        ``from_address`` overrides the submitter address (useful in
        tests).
        """
        with self._lock:
            payload = close_result.get("authoritative_payload")
            if payload is None:
                return AnchorResult(
                    success=False,
                    error="close_result has no authoritative_payload",
                )

            epoch_id = str(close_result.get("epoch_id") or "")
            if not epoch_id:
                # Fall back to a deterministic id derived from the root.
                epoch_id = f"e_{payload.get('epoch_root', '')[:16]}"
            epoch_root_hex = str(payload.get("epoch_root", ""))
            if not epoch_root_hex:
                return AnchorResult(
                    success=False,
                    error="authoritative_payload missing epoch_root",
                )

            prev_root_bytes = self.latest_epoch_root()
            prev_anchor_bytes = self.latest_anchor_hash()
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
                return AnchorResult(success=False, error=f"encoding failed: {e}")

            ph = payload_hash(payload_bytes)

            # Encode + CID the off-chain agent_mint blob.
            agent_mint = close_result.get("agent_mint") or payload.get("agent_mint", {})
            mint_blob = encode_agent_mint_blob(agent_mint)
            cid = cid_for_blob(mint_blob)

            # Merkle root over the mint map: agents prove their share
            # against this when claiming (recordTrainingForEpoch).
            mint_root = mint_merkle_root(agent_mint)

            try:
                tx_hash = self._submit(
                    epoch_id=epoch_id,
                    epoch_root=bytes.fromhex(epoch_root_hex),
                    prev_epoch_root=prev_root_bytes,
                    prev_anchor_hash=prev_anchor_bytes,
                    agent_mint_cid=cid,
                    payload_hash_bytes=ph,
                    agent_mint_root=mint_root,
                    from_address=from_address,
                )
            except Exception as e:
                return AnchorResult(
                    success=False,
                    error=f"submitAnchor reverted: {e}",
                    epoch_id=epoch_id,
                    epoch_root_hex=epoch_root_hex,
                    prev_epoch_root_hex=prev_root_hex,
                    agent_mint_cid=cid,
                    agent_mint_blob=mint_blob,
                    payload_bytes=payload_bytes,
                    payload_hash_hex=ph.hex(),
                    agent_mint_root_hex=mint_root.hex(),
                )

            return AnchorResult(
                success=True,
                tx_hash=tx_hash,
                epoch_id=epoch_id,
                epoch_root_hex=epoch_root_hex,
                prev_epoch_root_hex=prev_root_hex,
                agent_mint_cid=cid,
                agent_mint_blob=mint_blob,
                payload_bytes=payload_bytes,
                payload_hash_hex=ph.hex(),
                agent_mint_root_hex=mint_root.hex(),
            )

    def _submit(
        self,
        *,
        epoch_id: str,
        epoch_root: bytes,
        prev_epoch_root: bytes,
        prev_anchor_hash: bytes,
        agent_mint_cid: str,
        payload_hash_bytes: bytes,
        agent_mint_root: bytes,
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
            tx_hash = contract.functions.submitAnchor(
                epoch_id,
                epoch_root,
                prev_epoch_root,
                prev_anchor_hash,
                agent_mint_cid,
                payload_hash_bytes,
                agent_mint_root,
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
        # Etherlink shadownet reports ~1.2M for submitAnchor). A flat
        # hardcode bites in real-world deployments.
        try:
            estimated = contract.functions.submitAnchor(
                epoch_id,
                epoch_root,
                prev_epoch_root,
                prev_anchor_hash,
                agent_mint_cid,
                payload_hash_bytes,
                agent_mint_root,
            ).estimate_gas({"from": sender})
            gas_limit = max(int(estimated * 12 // 10), 1_500_000)
        except Exception:
            gas_limit = 2_500_000
        tx = contract.functions.submitAnchor(
            epoch_id,
            epoch_root,
            prev_epoch_root,
            prev_anchor_hash,
            agent_mint_cid,
            payload_hash_bytes,
            agent_mint_root,
        ).build_transaction({
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
        logger.info("Anchor submitted: epoch=%s, tx=%s", epoch_id, tx_hash.hex())
        return tx_hash.hex()
