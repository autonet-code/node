"""On-chain agent registration service for RPB contracts.

Handles building and signing registerAgent transactions against the
RPB contract. For root agents, returns unsigned call data so the
frontend wallet can sign. For child agents, the daemon holds the
private key and signs directly.
"""
from __future__ import annotations

import hashlib
import logging
import struct
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RPBConfig
    from .models import AgentIdentity

log = logging.getLogger(__name__)

# Minimal ABI for the functions we call — avoids loading full artifact
RPB_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "lineageHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "alignmentHash", "type": "bytes32"},
            {"internalType": "address", "name": "parentAgent", "type": "address"},
            {"internalType": "address", "name": "sponsorAgent", "type": "address"},
        ],
        "name": "registerAgent",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "newAlignmentHash", "type": "bytes32"}],
        "name": "updateAlignment",
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
        "name": "isActive",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "agents",
        "outputs": [
            {"internalType": "address", "name": "agentAddress", "type": "address"},
            {"internalType": "bytes32", "name": "lineageHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "alignmentHash", "type": "bytes32"},
            {"internalType": "address", "name": "parentAgent", "type": "address"},
            {"internalType": "address", "name": "sponsorAgent", "type": "address"},
            {"internalType": "uint256", "name": "registeredAt", "type": "uint256"},
            {"internalType": "bool", "name": "active", "type": "bool"},
            {"internalType": "uint256", "name": "totalInferenceServed", "type": "uint256"},
            {"internalType": "uint256", "name": "totalTrainingContributions", "type": "uint256"},
            {"internalType": "uint256", "name": "totalRewardsEarned", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

RPB_CONSTITUTION_ABI = [
    {
        "inputs": [],
        "name": "constitution",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getConstitution",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
]

REGISTRY_ABI = [
    {
        "inputs": [{"internalType": "string", "name": "key", "type": "string"}],
        "name": "getRegistryValue",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Additional RPB view functions for training/alignment queries
RPB_VIEWS_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "sponsor", "type": "address"}],
        "name": "getSponsorBudgetRemaining",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "isSponsor",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "getTrainingTokens",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "getUnclaimedRewards",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "currentEpoch",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "trainingRewardPool",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalTrainingTokens",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalInferenceUnits",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalInferenceRevenue",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "currentModelHash",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "modelVersion",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getExchangeRate",
        "outputs": [
            {"internalType": "uint256", "name": "numerator", "type": "uint256"},
            {"internalType": "uint256", "name": "denominator", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "holder", "type": "address"}],
        "name": "getShareBalance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "holder", "type": "address"}],
        "name": "getUnclaimedDividends",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "token", "type": "address"}],
        "name": "getTokenParity",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "token", "type": "address"}],
        "name": "getTokenPoolBalance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalShares",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "getChildren",
        "outputs": [{"internalType": "address[]", "name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getRegisteredAgents",
        "outputs": [{"internalType": "address[]", "name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Event ABI for replaying AgentRegistered logs
AGENT_REGISTERED_EVENT_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "agent", "type": "address"},
            {"indexed": False, "internalType": "bytes32", "name": "lineageHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "parentAgent", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "sponsor", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "name": "AgentRegistered",
        "type": "event",
    },
]

# Write functions for sponsorship/inference pipeline
RPB_SPONSOR_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "charterHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "initialBudget", "type": "uint256"},
        ],
        "name": "registerAsSponsor",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "newBudget", "type": "uint256"}],
        "name": "updateSponsorBudget",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "requester", "type": "address"},
            {"internalType": "address", "name": "provider", "type": "address"},
            {"internalType": "uint256", "name": "units", "type": "uint256"},
            {"internalType": "address", "name": "token", "type": "address"},
            {"internalType": "uint256", "name": "cost", "type": "uint256"},
        ],
        "name": "recordInference",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

# Write functions for investment/funding pipeline
RPB_INVEST_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "token", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "purchaseShares",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "token", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "fundTrainingPool",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "token", "type": "address"}],
        "name": "claimTrainingReward",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "token", "type": "address"}],
        "name": "claimDividends",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

GOVERNOR_ABI = [
    {
        "inputs": [],
        "name": "token",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "timelock",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "name",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
]

REPTOKEN_ABI = [
    {
        "inputs": [],
        "name": "registryAddress",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "economyAddress",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
]

REGISTRY_KEYS_ABI = [
    {
        "inputs": [{"internalType": "string", "name": "key", "type": "string"}],
        "name": "getRegistryValue",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getAllKeys",
        "outputs": [{"internalType": "string[]", "name": "", "type": "string[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getAllValues",
        "outputs": [{"internalType": "string[]", "name": "", "type": "string[]"}],
        "stateMutability": "view",
        "type": "function",
    },
]

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def discover_jurisdiction(rpc_url: str, dao_address: str) -> dict[str, str]:
    """Discover all jurisdiction contract addresses from the DAO Governor.

    Chain: Governor -> token() -> RepToken -> registryAddress() -> Registry
                                           -> economyAddress() -> Economy
           Governor -> timelock() -> Timelock
           Registry -> getRegistryValue("rpb.contract") -> RPB

    Returns a dict with keys: dao, token, registry, economy, timelock, rpb,
    jurisdiction_name, and any registry key-value pairs.
    """
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to {rpc_url}")

    dao = w3.to_checksum_address(dao_address)
    governor = w3.eth.contract(address=dao, abi=GOVERNOR_ABI)

    name = governor.functions.name().call()
    token_addr = governor.functions.token().call()
    timelock_addr = governor.functions.timelock().call()

    token = w3.eth.contract(address=token_addr, abi=REPTOKEN_ABI)
    registry_addr = token.functions.registryAddress().call()
    economy_addr = token.functions.economyAddress().call()

    registry = w3.eth.contract(address=registry_addr, abi=REGISTRY_KEYS_ABI)
    rpb_addr = ""
    try:
        rpb_addr = registry.functions.getRegistryValue("rpb.contract").call()
    except Exception:
        pass

    result = {
        "dao": dao,
        "token": token_addr,
        "registry": registry_addr,
        "economy": economy_addr,
        "timelock": timelock_addr,
        "rpb": rpb_addr,
        "jurisdiction_name": name,
    }

    # Load all registry key-value pairs
    try:
        keys = registry.functions.getAllKeys().call()
        values = registry.functions.getAllValues().call()
        for k, v in zip(keys, values):
            result[f"registry.{k}"] = v
    except Exception:
        pass

    log.info("Discovered jurisdiction '%s': %s", name, {
        k: v for k, v in result.items() if not k.startswith("registry.")
    })
    return result


def _to_bytes32(hex_str: str) -> bytes:
    """Convert a hex string (with or without 0x) to 32 bytes."""
    clean = hex_str.replace("0x", "")
    return bytes.fromhex(clean.ljust(64, "0")[:64])


def compute_alignment_hash(system_prompt: str) -> str:
    """Compute alignment hash from the agent's system prompt.

    Produces a bytes32-compatible hex digest by:
    1. Computing a semantic embedding of the prompt via
       ``nodes.common.alignment_pricing.compute_text_embedding``.
    2. Serializing the float vector to bytes (IEEE 754 doubles).
    3. SHA256-hashing those bytes.

    Falls back to a plain SHA256 of the prompt text when the embedding
    library is unavailable (e.g. torch not installed).
    """
    try:
        from nodes.common.alignment_pricing import compute_text_embedding

        embedding: list[float] = compute_text_embedding(system_prompt)
        # Pack every float as a little-endian double (8 bytes each)
        embedding_bytes = struct.pack(f"<{len(embedding)}d", *embedding)
        return hashlib.sha256(embedding_bytes).hexdigest()
    except Exception:
        # Fallback: content hash when embedding pipeline is unavailable
        log.debug(
            "Semantic embedding unavailable, falling back to content hash"
        )
        return hashlib.sha256(system_prompt.encode()).hexdigest()


class OnChainService:
    """On-chain agent registration and RPB queries."""

    def __init__(self, config: RPBConfig) -> None:
        self.config = config

    @property
    def available(self) -> bool:
        """Whether on-chain operations are possible."""
        return bool(
            self.config.rpb_contract_address
            and self.config.rpc_url
        )

    def _get_web3(self):
        """Get a connected Web3 instance."""
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
        if not w3.is_connected():
            raise ConnectionError(f"Cannot connect to {self.config.rpc_url}")
        return w3

    def _get_rpb_contract(self, w3=None):
        """Get the RPB contract instance."""
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        return w3.eth.contract(
            address=Web3.to_checksum_address(self.config.rpb_contract_address),
            abi=RPB_ABI,
        )

    def _get_rpb_extended(self, w3=None):
        """Get RPB contract with extended view ABI (training, alignment, model)."""
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        return w3.eth.contract(
            address=Web3.to_checksum_address(self.config.rpb_contract_address),
            abi=RPB_ABI + RPB_VIEWS_ABI,
        )

    def _get_rpb_invest(self, w3=None):
        """Get RPB contract with investment write ABI."""
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        return w3.eth.contract(
            address=Web3.to_checksum_address(self.config.rpb_contract_address),
            abi=RPB_INVEST_ABI,
        )

    def _get_rpb_sponsor(self, w3=None):
        """Get RPB contract with sponsorship/inference write ABI."""
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        return w3.eth.contract(
            address=Web3.to_checksum_address(self.config.rpb_contract_address),
            abi=RPB_SPONSOR_ABI,
        )

    def _get_registry_contract(self, w3=None):
        """Get the jurisdiction Registry contract instance."""
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        if not self.config.registry_address:
            return None
        return w3.eth.contract(
            address=Web3.to_checksum_address(self.config.registry_address),
            abi=REGISTRY_ABI,
        )

    def build_register_call_data(
        self,
        identity: AgentIdentity,
        system_prompt: str = "",
        parent_address: str = "",
        sponsor_address: str = "",
    ) -> str:
        """Build the ABI-encoded call data for registerAgent.

        Returns hex-encoded call data (with 0x prefix) that the frontend
        wallet can use to send the transaction.
        """
        w3 = self._get_web3()
        contract = self._get_rpb_contract(w3)

        lineage_bytes = _to_bytes32(identity.lineage_hash)
        alignment_hash = compute_alignment_hash(system_prompt)
        alignment_bytes = _to_bytes32(alignment_hash)
        parent = w3.to_checksum_address(parent_address) if parent_address else ZERO_ADDRESS
        sponsor = w3.to_checksum_address(sponsor_address) if sponsor_address else ZERO_ADDRESS

        call_data = contract.functions.registerAgent(
            lineage_bytes,
            alignment_bytes,
            parent,
            sponsor,
        )._encode_transaction_data()

        return call_data

    def build_update_alignment_call_data(
        self,
        system_prompt: str,
    ) -> str:
        """Build ABI-encoded call data for updateAlignment(bytes32).

        Computes the semantic alignment hash from the given system prompt
        and returns hex-encoded call data (with 0x prefix) that a wallet
        can use to send the transaction.
        """
        w3 = self._get_web3()
        contract = self._get_rpb_contract(w3)

        alignment_hash = compute_alignment_hash(system_prompt)
        alignment_bytes = _to_bytes32(alignment_hash)

        call_data = contract.functions.updateAlignment(
            alignment_bytes,
        )._encode_transaction_data()

        return call_data

    async def update_alignment(
        self,
        private_key: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Update an agent's alignment hash on-chain.

        Computes the semantic alignment hash from the system prompt and
        submits an ``updateAlignment(bytes32)`` transaction.  For root
        agents that need wallet signing, use
        :meth:`build_update_alignment_call_data` instead.

        Args:
            private_key: Hex private key for the agent's wallet.
            system_prompt: The agent's current system prompt.

        Returns:
            dict with ``success``, ``tx_hash``, or ``error``.
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_rpb_contract(w3)
            account = Account.from_key(private_key)

            alignment_hash = compute_alignment_hash(system_prompt)
            alignment_bytes = _to_bytes32(alignment_hash)

            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id

            tx = contract.functions.updateAlignment(
                alignment_bytes,
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 200_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                log.info(
                    "Alignment updated on-chain: %s tx=%s",
                    account.address,
                    tx_hash.hex(),
                )
                return {
                    "success": True,
                    "tx_hash": tx_hash.hex(),
                    "alignment_hash": alignment_hash,
                }
            else:
                return {
                    "success": False,
                    "error": "Transaction reverted",
                    "tx_hash": tx_hash.hex(),
                }

        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to update alignment on-chain")
            return {"success": False, "error": str(e)}

    async def register_agent(
        self,
        identity: AgentIdentity,
        private_key: str,
        system_prompt: str = "",
        parent_address: str = "",
        sponsor_address: str = "",
    ) -> dict[str, Any]:
        """Register an agent on-chain by signing the transaction directly.

        Used for non-root agents where the daemon holds the private key.
        For root agents, use build_register_call_data() and let the
        frontend wallet sign.

        Args:
            identity: Agent's identity (lineage hash, address, etc.)
            private_key: Hex private key for signing
            system_prompt: Agent's system prompt (for alignment hash)
            parent_address: Parent agent's on-chain address (0x0 for root)
            sponsor_address: Sponsor agent's on-chain address (0x0 if none)

        Returns:
            dict with success, tx_hash, or error
        """
        try:
            from web3 import Web3
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_rpb_contract(w3)
            account = Account.from_key(private_key)

            lineage_bytes = _to_bytes32(identity.lineage_hash)
            alignment_hash = compute_alignment_hash(system_prompt)
            alignment_bytes = _to_bytes32(alignment_hash)
            parent = w3.to_checksum_address(parent_address) if parent_address else ZERO_ADDRESS
            sponsor = w3.to_checksum_address(sponsor_address) if sponsor_address else ZERO_ADDRESS

            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id

            tx = contract.functions.registerAgent(
                lineage_bytes,
                alignment_bytes,
                parent,
                sponsor,
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 500_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                log.info("Agent registered on-chain: %s tx=%s",
                         account.address, tx_hash.hex())
                return {
                    "success": True,
                    "tx_hash": tx_hash.hex(),
                    "agent_address": account.address,
                }
            else:
                return {
                    "success": False,
                    "error": "Transaction reverted",
                    "tx_hash": tx_hash.hex(),
                }

        except ImportError:
            return {"success": False, "error": "web3 package not installed"}
        except Exception as e:
            log.exception("Failed to register agent on-chain")
            return {"success": False, "error": str(e)}

    async def is_registered(self, address: str) -> bool:
        """Check if an agent is registered on-chain."""
        try:
            w3 = self._get_web3()
            contract = self._get_rpb_contract(w3)
            return contract.functions.isRegistered(
                w3.to_checksum_address(address)
            ).call()
        except Exception:
            return False

    async def get_agent_record(self, address: str) -> dict[str, Any] | None:
        """Read an agent's on-chain record."""
        try:
            w3 = self._get_web3()
            contract = self._get_rpb_contract(w3)
            result = contract.functions.agents(
                w3.to_checksum_address(address)
            ).call()

            # result is a tuple matching the struct fields
            agent_addr, lineage, alignment, parent, sponsor, registered_at, active, \
                inference, training, rewards = result

            if registered_at == 0:
                return None

            return {
                "agent_address": agent_addr,
                "lineage_hash": lineage.hex(),
                "alignment_hash": alignment.hex(),
                "parent_agent": parent,
                "sponsor_agent": sponsor,
                "registered_at": registered_at,
                "active": active,
                "total_inference_served": inference,
                "total_training_contributions": training,
                "total_rewards_earned": rewards,
            }
        except Exception as e:
            log.debug("Failed to read agent record for %s: %s", address, e)
            return None

    async def get_constitution_cid(self) -> str | None:
        """Read the current constitution CID from the jurisdiction Registry."""
        try:
            w3 = self._get_web3()
            registry = self._get_registry_contract(w3)
            if registry is None:
                return None
            value = registry.functions.getRegistryValue("rpb.prompt.current").call()
            return value if value else None
        except Exception as e:
            log.debug("Failed to read constitution CID: %s", e)
            return None

    async def get_constitution_text(self) -> str | None:
        """Read the full constitution text stored on-chain in the RPB contract."""
        try:
            w3 = self._get_web3()
            addr = self.config.rpb_contract_address
            if not addr:
                return None
            contract = w3.eth.contract(
                address=w3.to_checksum_address(addr),
                abi=RPB_CONSTITUTION_ABI,
            )
            text = contract.functions.constitution().call()
            return text if text else None
        except Exception as e:
            log.debug("Failed to read constitution text from RPB: %s", e)
            return None

    async def get_rpb_state(self) -> dict[str, Any] | None:
        """Read global RPB contract state (epoch, pools, model)."""
        try:
            w3 = self._get_web3()
            rpb = self._get_rpb_extended(w3)
            epoch = rpb.functions.currentEpoch().call()
            reward_pool = rpb.functions.trainingRewardPool().call()
            total_training = rpb.functions.totalTrainingTokens().call()
            total_inference = rpb.functions.totalInferenceUnits().call()
            total_revenue = rpb.functions.totalInferenceRevenue().call()
            model_hash = rpb.functions.currentModelHash().call()
            model_version = rpb.functions.modelVersion().call()
            num, denom = rpb.functions.getExchangeRate().call()
            return {
                "current_epoch": epoch,
                "training_reward_pool": str(reward_pool),
                "total_training_tokens": str(total_training),
                "total_inference_units": str(total_inference),
                "total_inference_revenue": str(total_revenue),
                "current_model_hash": model_hash.hex() if isinstance(model_hash, bytes) else str(model_hash),
                "model_version": model_version,
                "exchange_rate_numerator": str(num),
                "exchange_rate_denominator": str(denom),
            }
        except Exception as e:
            log.debug("Failed to read RPB state: %s", e)
            return None

    async def get_agent_training_info(self, address: str) -> dict[str, Any] | None:
        """Read an agent's training-specific on-chain data."""
        try:
            w3 = self._get_web3()
            rpb = self._get_rpb_extended(w3)
            addr = w3.to_checksum_address(address)
            tokens = rpb.functions.getTrainingTokens(addr).call()
            unclaimed = rpb.functions.getUnclaimedRewards(addr).call()
            record = await self.get_agent_record(address)
            return {
                "training_tokens": str(tokens),
                "unclaimed_rewards": str(unclaimed),
                "total_training_contributions": str(record["total_training_contributions"]) if record else "0",
                "total_rewards_earned": str(record["total_rewards_earned"]) if record else "0",
                "total_inference_served": str(record["total_inference_served"]) if record else "0",
            }
        except Exception as e:
            log.debug("Failed to read training info for %s: %s", address, e)
            return None

    # ------------------------------------------------------------------
    # Investment / funding pipeline
    # ------------------------------------------------------------------

    def build_purchase_shares_call_data(self, token_address: str, amount: int) -> str:
        """Build ABI-encoded call data for purchaseShares(token, amount).

        Returns hex-encoded call data (with 0x prefix) that the frontend
        wallet can use to send the transaction.
        """
        w3 = self._get_web3()
        contract = self._get_rpb_invest(w3)
        token = w3.to_checksum_address(token_address)
        return contract.functions.purchaseShares(token, amount)._encode_transaction_data()

    def build_fund_training_call_data(self, token_address: str, amount: int) -> str:
        """Build ABI-encoded call data for fundTrainingPool(token, amount)."""
        w3 = self._get_web3()
        contract = self._get_rpb_invest(w3)
        token = w3.to_checksum_address(token_address)
        return contract.functions.fundTrainingPool(token, amount)._encode_transaction_data()

    def build_claim_training_reward_call_data(self, token_address: str) -> str:
        """Build ABI-encoded call data for claimTrainingReward(token)."""
        w3 = self._get_web3()
        contract = self._get_rpb_invest(w3)
        token = w3.to_checksum_address(token_address)
        return contract.functions.claimTrainingReward(token)._encode_transaction_data()

    def build_claim_dividends_call_data(self, token_address: str) -> str:
        """Build ABI-encoded call data for claimDividends(token)."""
        w3 = self._get_web3()
        contract = self._get_rpb_invest(w3)
        token = w3.to_checksum_address(token_address)
        return contract.functions.claimDividends(token)._encode_transaction_data()

    async def get_investor_info(self, address: str) -> dict[str, Any] | None:
        """Read an investor's share balance and unclaimed dividends."""
        try:
            w3 = self._get_web3()
            rpb = self._get_rpb_extended(w3)
            addr = w3.to_checksum_address(address)
            share_balance = rpb.functions.getShareBalance(addr).call()
            unclaimed_dividends = rpb.functions.getUnclaimedDividends(addr).call()
            total_shares = rpb.functions.totalShares().call()
            return {
                "address": address,
                "share_balance": str(share_balance),
                "unclaimed_dividends": str(unclaimed_dividends),
                "total_shares": str(total_shares),
            }
        except Exception as e:
            log.debug("Failed to read investor info for %s: %s", address, e)
            return None

    async def get_accepted_tokens(self) -> list[dict]:
        """Discover accepted tokens by reading Registry keys matching jurisdiction.parity.*.

        Returns a list of dicts with token address, parity value, symbol, and name.
        """
        try:
            from web3 import Web3

            NATIVE = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
            ERC20_METADATA_ABI = [
                {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
                {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
            ]

            w3 = self._get_web3()
            if not self.config.registry_address:
                return []
            registry = w3.eth.contract(
                address=Web3.to_checksum_address(self.config.registry_address),
                abi=REGISTRY_KEYS_ABI,
            )
            keys = registry.functions.getAllKeys().call()
            values = registry.functions.getAllValues().call()

            prefix = "jurisdiction.parity."
            tokens = []
            for k, v in zip(keys, values):
                if k.startswith(prefix):
                    token_addr = k[len(prefix):]
                    symbol = ""
                    name = ""
                    decimals = 18
                    if token_addr.lower() == NATIVE:
                        symbol = "XTZ"
                        name = "Native Currency"
                    else:
                        try:
                            erc20 = w3.eth.contract(
                                address=Web3.to_checksum_address(token_addr),
                                abi=ERC20_METADATA_ABI,
                            )
                            symbol = erc20.functions.symbol().call()
                            name = erc20.functions.name().call()
                            decimals = erc20.functions.decimals().call()
                        except Exception:
                            symbol = f"{token_addr[:6]}...{token_addr[-4:]}"
                            name = "Unknown Token"
                    tokens.append({
                        "token_address": token_addr,
                        "parity": v,
                        "symbol": symbol,
                        "name": name,
                        "decimals": decimals,
                        "registry_key": k,
                    })
            return tokens
        except Exception as e:
            log.debug("Failed to read accepted tokens: %s", e)
            return []

    # ------------------------------------------------------------------
    # Sponsorship / inference pipeline
    # ------------------------------------------------------------------

    async def record_inference(
        self,
        requester: str,
        provider: str,
        units: int,
        token_address: str,
        cost: int,
        private_key: str,
    ) -> dict[str, Any]:
        """Record an inference event on-chain via recordInference.

        This is an ``onlyOwner`` function on RPB, so ``private_key`` must
        be the deployer/owner key (from config.private_key).

        Args:
            requester: Address of the agent that requested inference.
            provider: Address of the agent that served inference.
            units: Number of inference units consumed.
            token_address: ERC20 payment token (must be in value index).
            cost: Raw token amount for the inference cost.
            private_key: Owner's hex private key for signing.

        Returns:
            dict with ``success``, ``tx_hash``, or ``error``.
        """
        try:
            from web3 import Web3
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_rpb_sponsor(w3)
            account = Account.from_key(private_key)

            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id

            tx = contract.functions.recordInference(
                w3.to_checksum_address(requester),
                w3.to_checksum_address(provider),
                units,
                w3.to_checksum_address(token_address),
                cost,
            ).build_transaction({
                "from": account.address,
                "nonce": nonce,
                "gas": 300_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                log.info(
                    "Inference recorded on-chain: requester=%s provider=%s units=%d tx=%s",
                    requester, provider, units, tx_hash.hex(),
                )
                return {
                    "success": True,
                    "tx_hash": tx_hash.hex(),
                    "units": units,
                    "cost": cost,
                }
            else:
                return {
                    "success": False,
                    "error": "Transaction reverted",
                    "tx_hash": tx_hash.hex(),
                }

        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to record inference on-chain")
            return {"success": False, "error": str(e)}

    async def get_sponsor_budget(self, address: str) -> dict[str, Any]:
        """Read a sponsor's remaining budget from the RPB contract.

        Calls ``getSponsorBudgetRemaining(address)`` and ``isSponsor(address)``.

        Returns:
            dict with ``address``, ``is_sponsor``, ``budget_remaining``
            (normalized value units as string), or ``error``.
        """
        try:
            w3 = self._get_web3()
            rpb = self._get_rpb_extended(w3)
            addr = w3.to_checksum_address(address)
            is_sponsor = rpb.functions.isSponsor(addr).call()
            remaining = rpb.functions.getSponsorBudgetRemaining(addr).call()
            return {
                "address": address,
                "is_sponsor": is_sponsor,
                "budget_remaining": str(remaining),
            }
        except Exception as e:
            log.debug("Failed to read sponsor budget for %s: %s", address, e)
            return {"address": address, "error": str(e)}

    def build_register_as_sponsor_call_data(
        self, charter_hash: str, initial_budget: int,
    ) -> str:
        """Build ABI-encoded call data for registerAsSponsor(bytes32, uint256).

        Returns hex-encoded call data (with 0x prefix) for wallet signing.
        """
        w3 = self._get_web3()
        contract = self._get_rpb_sponsor(w3)
        charter_bytes = _to_bytes32(charter_hash)
        return contract.functions.registerAsSponsor(
            charter_bytes, initial_budget,
        )._encode_transaction_data()

    def build_update_sponsor_budget_call_data(self, new_budget: int) -> str:
        """Build ABI-encoded call data for updateSponsorBudget(uint256).

        Returns hex-encoded call data (with 0x prefix) for wallet signing.
        """
        w3 = self._get_web3()
        contract = self._get_rpb_sponsor(w3)
        return contract.functions.updateSponsorBudget(
            new_budget,
        )._encode_transaction_data()

    # ------------------------------------------------------------------
    # Agent enumeration (event replay)
    # ------------------------------------------------------------------

    async def get_all_registered_agents(self) -> list[dict[str, Any]]:
        """Enumerate all registered agents via getRegisteredAgents() view.

        Returns:
            List of dicts with agent record fields plus ``is_sponsor``
            and ``children_count``.
        """
        try:
            from web3 import Web3

            w3 = self._get_web3()
            rpb_addr = Web3.to_checksum_address(self.config.rpb_contract_address)
            rpb = w3.eth.contract(
                address=rpb_addr,
                abi=RPB_ABI + RPB_VIEWS_ABI,
            )

            # Get all agent addresses in one call
            all_addresses = rpb.functions.getRegisteredAgents().call()
            if not all_addresses:
                return []

            agents: list[dict[str, Any]] = []
            for agent_addr in all_addresses:
                addr_cs = Web3.to_checksum_address(agent_addr)
                try:
                    record = rpb.functions.agents(addr_cs).call()
                    (r_addr, lineage, alignment, parent, sponsor,
                     registered_at, active, inference, training, rewards) = record

                    is_sponsor = rpb.functions.isSponsor(addr_cs).call()
                    children = rpb.functions.getChildren(addr_cs).call()

                    agents.append({
                        "agent_address": r_addr,
                        "lineage_hash": lineage.hex() if isinstance(lineage, bytes) else str(lineage),
                        "alignment_hash": alignment.hex() if isinstance(alignment, bytes) else str(alignment),
                        "parent_agent": parent,
                        "sponsor_agent": sponsor,
                        "registered_at": registered_at,
                        "active": active,
                        "total_inference_served": str(inference),
                        "total_training_contributions": str(training),
                        "total_rewards_earned": str(rewards),
                        "is_sponsor": is_sponsor,
                        "children_count": len(children),
                    })
                except Exception as e:
                    log.debug("Failed to read record for agent %s: %s", agent_addr, e)
                    continue

            return agents

        except Exception as e:
            log.debug("Failed to enumerate registered agents: %s", e)
            return []
