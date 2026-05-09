"""On-chain agent registration + reads against ``Substrate.sol``.

This module is the daemon's interface to the substrate-native chain
surface. It supersedes the pre-substrate ``RPB`` contract integration
(which targeted a contract that no longer exists post-Phase 5.6a nuke).

What lives here
---------------

- ``register_agent`` / ``build_register_call_data`` — register an
  agent with a libp2p PeerId so daemons can DHT-resolve it later.
- ``update_peer_id`` / ``build_update_peer_id_call_data`` — rotate
  the libp2p keypair without losing on-chain identity.
- ``is_registered``, ``get_agent_record`` — basic membership reads.
- ``get_agent_peer_id`` — for off-chain peer discovery.
- ``get_all_registered_agents`` — enumerate agents (for discovery).
- ``get_agent_balances`` — reputation (soulbound) + ATN (transferable)
  + per-epoch mint history for an agent.
- ``get_anchor_count`` / ``get_anchor`` — read epoch anchors directly.

What was removed
----------------

The pre-substrate RPB surface (parent/sponsor agents, alignment hashes,
share purchases, dividend claims, sponsor budgets, training reward pools,
exchange rates, recordInference) is gone. Those concepts were jurisdiction-
shaped fragments that didn't survive the substrate rewrite. The frontend
will need substrate-native equivalents — see ws_server stubs.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RPBConfig
    from .models import AgentIdentity

log = logging.getLogger(__name__)


# ABI for Substrate.sol — only the surface we actually call. Kept inline
# so on_chain.py doesn't need the artifact JSON to load.
SUBSTRATE_ABI = [
    # Agent registration / discovery
    {
        "inputs": [
            {"internalType": "bytes32", "name": "lineageHash", "type": "bytes32"},
            {"internalType": "bytes",   "name": "peerId",      "type": "bytes"},
        ],
        "name": "registerAgent",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes", "name": "peerId", "type": "bytes"}],
        "name": "updatePeerId",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "isRegistered",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "getAgentPeerId",
        "outputs": [{"internalType": "bytes", "name": "", "type": "bytes"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "agents",
        "outputs": [
            {"internalType": "bytes32", "name": "lineageHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "registeredAt", "type": "uint256"},
            {"internalType": "bool",    "name": "active", "type": "bool"},
            {"internalType": "uint256", "name": "totalTrainingMint", "type": "uint256"},
            {"internalType": "uint256", "name": "trainingSubmissionCount", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "registeredAgentCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "name": "getRegisteredAgent",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    # Per-agent balances
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "agentReputation",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "", "type": "address"},
            {"internalType": "bytes32", "name": "", "type": "bytes32"},
        ],
        "name": "mintForEpoch",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    # Anchor reads
    {
        "inputs": [],
        "name": "anchorCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "latestAnchorHash",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "latestEpochRoot",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "atnTotalSupply",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "networkMintTotal",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    # Events used for log replay
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "agent",       "type": "address"},
            {"indexed": True,  "internalType": "bytes32", "name": "lineageHash", "type": "bytes32"},
            {"indexed": False, "internalType": "bytes",   "name": "peerId",      "type": "bytes"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp",   "type": "uint256"},
        ],
        "name": "AgentRegistered",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "agent",  "type": "address"},
            {"indexed": False, "internalType": "bytes",   "name": "peerId", "type": "bytes"},
        ],
        "name": "PeerIdUpdated",
        "type": "event",
    },
]


ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def _to_bytes32(hex_str: str) -> bytes:
    """Convert a hex string (with or without 0x) to 32 bytes."""
    clean = hex_str.replace("0x", "")
    return bytes.fromhex(clean.ljust(64, "0")[:64])


def _to_peer_id_bytes(peer_id: str | bytes) -> bytes:
    """Coerce a peer id to the bytes shape Substrate.sol expects.

    Accepts either raw ``bytes`` (already in libp2p multihash form) or
    a string (base58-encoded PeerId, e.g. ``12D3KooW...``). Strings get
    decoded via base58; bytes pass through.
    """
    if isinstance(peer_id, bytes):
        return peer_id
    if not peer_id:
        raise ValueError("peer_id is required")
    # base58 decode for libp2p PeerId strings.
    try:
        import base58  # type: ignore
        return base58.b58decode(peer_id)
    except ImportError:
        # libp2p ships base58 as a transitive dep, so this is rare. Fall
        # back to UTF-8 bytes — only correct for synthetic test ids.
        return peer_id.encode("utf-8")


class OnChainService:
    """Daemon-side helper for Substrate.sol reads + writes.

    Mirrors the pre-substrate ``OnChainService`` shape so the
    ws_server handlers don't need to be rewired beyond the few
    methods that disappeared with the RPB contract.
    """

    def __init__(self, config: "RPBConfig") -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether on-chain operations are possible.

        Reads from ``substrate_address`` if set; falls back to the
        legacy ``rpb_contract_address`` field for back-compat with
        configs that haven't migrated yet.
        """
        addr = self._substrate_address()
        return bool(addr and self.config.rpc_url)

    def _substrate_address(self) -> str:
        """Resolve the deployed Substrate.sol address from config."""
        addr = getattr(self.config, "substrate_address", "") or ""
        if not addr:
            # Legacy field name still used in many configs.
            addr = getattr(self.config, "rpb_contract_address", "") or ""
        return addr

    def _get_web3(self):
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
        if not w3.is_connected():
            raise ConnectionError(f"Cannot connect to {self.config.rpc_url}")
        return w3

    def _get_contract(self, w3=None):
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        return w3.eth.contract(
            address=Web3.to_checksum_address(self._substrate_address()),
            abi=SUBSTRATE_ABI,
        )

    # ------------------------------------------------------------------
    # Agent registration (write)
    # ------------------------------------------------------------------

    def build_register_call_data(
        self,
        identity: "AgentIdentity",
        peer_id: str | bytes,
        # Kept for back-compat with ws_server handler signatures, ignored:
        system_prompt: str = "",
        parent_address: str = "",
        sponsor_address: str = "",
    ) -> str:
        """Build ABI-encoded call data for ``registerAgent(lineage, peerId)``.

        Returns hex-encoded call data (with 0x prefix) that the
        frontend wallet can use to send the transaction. ``system_prompt``,
        ``parent_address``, ``sponsor_address`` are accepted for
        legacy callsite stability and ignored (the substrate has no
        analog).
        """
        contract = self._get_contract()
        lineage_bytes = _to_bytes32(identity.lineage_hash)
        peer_bytes = _to_peer_id_bytes(peer_id)
        return contract.functions.registerAgent(
            lineage_bytes, peer_bytes,
        )._encode_transaction_data()

    async def register_agent(
        self,
        identity: "AgentIdentity",
        private_key: str,
        peer_id: str | bytes,
        # Back-compat parameters, ignored:
        system_prompt: str = "",
        parent_address: str = "",
        sponsor_address: str = "",
    ) -> dict[str, Any]:
        """Sign and submit a ``registerAgent`` transaction directly.

        Used when the daemon holds the private key. For wallet-signed
        registration, use :meth:`build_register_call_data` instead.
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)

            lineage_bytes = _to_bytes32(identity.lineage_hash)
            peer_bytes = _to_peer_id_bytes(peer_id)

            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id

            # Estimate gas with a margin for networks that report higher
            # actual costs (e.g. Etherlink shadownet).
            try:
                estimated = contract.functions.registerAgent(
                    lineage_bytes, peer_bytes,
                ).estimate_gas({"from": account.address})
                gas_limit = max(int(estimated * 12 // 10), 500_000)
            except Exception:
                gas_limit = 1_500_000

            tx = contract.functions.registerAgent(
                lineage_bytes, peer_bytes,
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": gas_limit,
                "gasPrice": w3.eth.gas_price,
                "chainId": chain_id,
            })

            signed = account.sign_transaction(tx)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            tx_hash = w3.eth.send_raw_transaction(raw)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                log.info(
                    "Agent registered on substrate: %s tx=%s",
                    account.address, tx_hash.hex(),
                )
                return {
                    "success": True,
                    "tx_hash": tx_hash.hex(),
                    "agent_address": account.address,
                }
            return {
                "success": False,
                "error": "Transaction reverted",
                "tx_hash": tx_hash.hex(),
            }
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to register agent on chain")
            return {"success": False, "error": str(e)}

    def build_update_peer_id_call_data(self, peer_id: str | bytes) -> str:
        """Build ABI-encoded call data for ``updatePeerId(peerId)``."""
        contract = self._get_contract()
        peer_bytes = _to_peer_id_bytes(peer_id)
        return contract.functions.updatePeerId(peer_bytes)._encode_transaction_data()

    async def update_peer_id(
        self,
        private_key: str,
        peer_id: str | bytes,
    ) -> dict[str, Any]:
        """Sign and submit an ``updatePeerId`` transaction."""
        try:
            from eth_account import Account
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)
            peer_bytes = _to_peer_id_bytes(peer_id)
            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id
            tx = contract.functions.updatePeerId(peer_bytes).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 200_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": chain_id,
            })
            signed = account.sign_transaction(tx)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            tx_hash = w3.eth.send_raw_transaction(raw)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt.status == 1:
                return {"success": True, "tx_hash": tx_hash.hex()}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hash.hex()}
        except Exception as e:
            log.exception("Failed to update peer id on chain")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def is_registered(self, address: str) -> bool:
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            return contract.functions.isRegistered(
                w3.to_checksum_address(address)
            ).call()
        except Exception:
            return False

    async def get_agent_record(self, address: str) -> dict[str, Any] | None:
        """Read an agent's on-chain record from Substrate.sol's ``agents`` map."""
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            addr = w3.to_checksum_address(address)
            result = contract.functions.agents(addr).call()
            lineage, registered_at, active, total_mint, submission_count = result
            if registered_at == 0:
                return None
            peer_id = contract.functions.getAgentPeerId(addr).call()
            return {
                "agent_address": addr,
                "lineage_hash": lineage.hex() if isinstance(lineage, bytes) else str(lineage),
                "peer_id": peer_id.hex() if isinstance(peer_id, bytes) else str(peer_id),
                "registered_at": registered_at,
                "active": active,
                "total_training_mint": str(total_mint),
                "training_submission_count": submission_count,
            }
        except Exception as e:
            log.debug("Failed to read agent record for %s: %s", address, e)
            return None

    async def get_agent_peer_id(self, address: str) -> bytes | None:
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            return contract.functions.getAgentPeerId(
                w3.to_checksum_address(address)
            ).call()
        except Exception:
            return None

    async def get_agent_balances(self, address: str) -> dict[str, Any] | None:
        """Read reputation + ATN balance for an agent.

        Reputation is the soulbound ``agentMintTotal`` (renamed
        semantically). ATN is the transferable ERC20-shaped balance
        on the same contract.
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            addr = w3.to_checksum_address(address)
            reputation = contract.functions.agentReputation(addr).call()
            atn = contract.functions.balanceOf(addr).call()
            return {
                "address": addr,
                "reputation": str(reputation),
                "atn_balance": str(atn),
            }
        except Exception as e:
            log.debug("Failed to read balances for %s: %s", address, e)
            return None

    async def get_all_registered_agents(self) -> list[dict[str, Any]]:
        """Enumerate registered agents and return their records.

        Walks ``registeredAgentCount`` + ``getRegisteredAgent(i)`` in
        order. For very large agent sets a future optimization could
        replay ``AgentRegistered`` event logs instead.
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            count = contract.functions.registeredAgentCount().call()
            out: list[dict[str, Any]] = []
            for i in range(count):
                try:
                    addr = contract.functions.getRegisteredAgent(i).call()
                    record = await self.get_agent_record(addr)
                    if record is not None:
                        out.append(record)
                except Exception as e:
                    log.debug("Failed to read agent %d: %s", i, e)
                    continue
            return out
        except Exception as e:
            log.debug("Failed to enumerate registered agents: %s", e)
            return []

    async def get_substrate_state(self) -> dict[str, Any] | None:
        """Read network-wide substrate state.

        Returns anchor count, latest anchor + epoch root, total ATN
        supply, and total minted reputation (network-wide).
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            anchor_count = contract.functions.anchorCount().call()
            latest_anchor = contract.functions.latestAnchorHash().call()
            latest_root = contract.functions.latestEpochRoot().call()
            atn_supply = contract.functions.atnTotalSupply().call()
            mint_total = contract.functions.networkMintTotal().call()
            agent_count = contract.functions.registeredAgentCount().call()
            return {
                "anchor_count": anchor_count,
                "latest_anchor_hash": latest_anchor.hex() if isinstance(latest_anchor, bytes) else str(latest_anchor),
                "latest_epoch_root": latest_root.hex() if isinstance(latest_root, bytes) else str(latest_root),
                "atn_total_supply": str(atn_supply),
                "network_mint_total": str(mint_total),
                "registered_agent_count": agent_count,
            }
        except Exception as e:
            log.debug("Failed to read substrate state: %s", e)
            return None
