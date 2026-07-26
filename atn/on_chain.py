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
    # Owner-rooted registration (docs/tool_substrate.md §Owner-rooted
    # registration). The AGENT (msg.sender) submits; the OWNER wallet has
    # signed an EIP-712 OwnerBinding{agent, parent, nonce} over the domain
    # ("AutonetSubstrate", "1", chainId, this contract). rotateOwner needs
    # NO old-owner sig (key-loss recovery); the per-agent bindingNonce closes
    # replay.
    {
        "inputs": [
            {"internalType": "bytes32", "name": "lineageHash",  "type": "bytes32"},
            {"internalType": "bytes",   "name": "peerId",       "type": "bytes"},
            {"internalType": "address", "name": "owner",        "type": "address"},
            {"internalType": "address", "name": "parentAgent",  "type": "address"},
            {"internalType": "bytes",   "name": "ownerSig",     "type": "bytes"},
        ],
        "name": "registerAgentBound",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "newOwner",    "type": "address"},
            {"internalType": "address", "name": "newParent",   "type": "address"},
            {"internalType": "bytes",   "name": "newOwnerSig", "type": "bytes"},
        ],
        "name": "rotateOwner",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "bindingNonce",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
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
        "inputs": [{"internalType": "address[]", "name": "addrs", "type": "address[]"}],
        "name": "areRegistered",
        "outputs": [{"internalType": "bool[]", "name": "", "type": "bool[]"}],
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
    # Browser-reachable wss endpoint — agent-signed presence, mirrored to the
    # off-chain directory by the indexer (EndpointUpdated event).
    {
        "inputs": [{"internalType": "string", "name": "wsEndpoint", "type": "string"}],
        "name": "updateEndpoint",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "getAgentEndpoint",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
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
    # Owner binding (agent -> owner wallet; zero address = unbound)
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "agentOwner",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    # Per-agent balances. Money only (2026-07-10): agentMintTotal is the
    # cumulative tool-pool earnings counter (the agentReputation view is
    # deleted from Substrate — REP is DAO-side on ratified earnings).
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "agentMintTotal",
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
    # Per-epoch training record (money only, Decision 2026-07-10).
    # ``amount`` = ATN mint, merkle-proven: the leaf is
    # keccak256(bytes.concat(keccak256(abi.encode(agent, amount)))) under
    # the anchor's agentMintRoot (built by nodes/common/mint_merkle.py).
    # REP is no longer minted here — it is claimed DAO-side (RepToken) on
    # ratified ATN earnings, so the leaf reverts to 2-field (agent, amount).
    {
        "inputs": [
            {"internalType": "uint256",   "name": "amount",      "type": "uint256"},
            {"internalType": "bytes32",   "name": "epochIdHash", "type": "bytes32"},
            {"internalType": "bytes32[]", "name": "proof",       "type": "bytes32[]"},
        ],
        "name": "recordTrainingForEpoch",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    # Service payments (Phase 7.2). A labeled ATN transfer taking the
    # service fee and recording serviceEarnings; pricing is off-chain.
    {
        "inputs": [
            {"internalType": "address", "name": "recipient", "type": "address"},
            {"internalType": "uint256", "name": "amount",    "type": "uint256"},
            {"internalType": "bytes32", "name": "requestId", "type": "bytes32"},
        ],
        "name": "payForService",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "serviceEarnings",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "serviceFeeBps",
        "outputs": [{"internalType": "uint16", "name": "", "type": "uint16"}],
        "stateMutability": "view",
        "type": "function",
    },
    # ServicePayment / ServiceFee events — decoded from a tx receipt by
    # verify_service_payment (no log-range scan; Etherlink caps ranges).
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "payer",     "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "recipient", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount",    "type": "uint256"},
            {"indexed": True,  "internalType": "bytes32", "name": "requestId", "type": "bytes32"},
        ],
        "name": "ServicePayment",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "payer",      "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "recipient",  "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount",     "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "burned",     "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "toTreasury", "type": "uint256"},
        ],
        "name": "ServiceFee",
        "type": "event",
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
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "agent",     "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "owner",     "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "parent",    "type": "address"},
            {"indexed": False, "internalType": "address", "name": "prevOwner", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "nonce",     "type": "uint256"},
        ],
        "name": "OwnerBound",
        "type": "event",
    },
]


# ABI for ServiceMarket.sol's ServiceRegistry — only the surface we call.
# Derived from contracts/core/ServiceMarket.sol (registerService/updateServiceAsk/
# retireService writes; serviceCount/services/getService reads; the three events).
SERVICE_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "specDigest", "type": "bytes32"},
            {"internalType": "uint256", "name": "askAmount",  "type": "uint256"},
        ],
        "name": "registerService",
        "outputs": [{"internalType": "uint256", "name": "serviceId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "serviceId", "type": "uint256"},
            {"internalType": "uint256", "name": "askAmount", "type": "uint256"},
        ],
        "name": "updateServiceAsk",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "serviceId", "type": "uint256"}],
        "name": "retireService",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "serviceCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    # Auto-generated public getter for `mapping(uint256 => Service) services`.
    # Returns the struct fields as a flat tuple (Solidity flattens structs in
    # public-mapping getters).
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "services",
        "outputs": [
            {"internalType": "address", "name": "provider",     "type": "address"},
            {"internalType": "bytes32", "name": "specDigest",   "type": "bytes32"},
            {"internalType": "uint256", "name": "askAmount",    "type": "uint256"},
            {"internalType": "bool",    "name": "active",       "type": "bool"},
            {"internalType": "uint256", "name": "registeredAt", "type": "uint256"},
            {"internalType": "uint256", "name": "updatedAt",    "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    # Explicit getService(id) returns the Service struct (tuple-typed).
    {
        "inputs": [{"internalType": "uint256", "name": "serviceId", "type": "uint256"}],
        "name": "getService",
        "outputs": [
            {
                "components": [
                    {"internalType": "address", "name": "provider",     "type": "address"},
                    {"internalType": "bytes32", "name": "specDigest",   "type": "bytes32"},
                    {"internalType": "uint256", "name": "askAmount",    "type": "uint256"},
                    {"internalType": "bool",    "name": "active",       "type": "bool"},
                    {"internalType": "uint256", "name": "registeredAt", "type": "uint256"},
                    {"internalType": "uint256", "name": "updatedAt",    "type": "uint256"},
                ],
                "internalType": "struct ServiceRegistry.Service",
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "uint256", "name": "serviceId",  "type": "uint256"},
            {"indexed": True,  "internalType": "address", "name": "provider",   "type": "address"},
            {"indexed": True,  "internalType": "bytes32", "name": "specDigest", "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "askAmount",  "type": "uint256"},
        ],
        "name": "ServiceRegistered",
        "type": "event",
    },
]


# ABI for ServiceMarket.sol's PaymentChannel — the surface we call. Derived
# from contracts/core/ServiceMarket.sol (openChannel/closeChannel/
# withdrawRemainder writes; voucherHash/channelCount/channels/getChannel reads;
# the three events).
PAYMENT_CHANNEL_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "provider", "type": "address"},
            {"internalType": "uint256", "name": "deposit",  "type": "uint256"},
        ],
        "name": "openChannel",
        "outputs": [{"internalType": "uint256", "name": "channelId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "channelId",        "type": "uint256"},
            {"internalType": "uint256", "name": "cumulativeAmount", "type": "uint256"},
        ],
        "name": "voucherHash",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "channelId",        "type": "uint256"},
            {"internalType": "uint256", "name": "cumulativeAmount", "type": "uint256"},
            {"internalType": "bytes",   "name": "signature",        "type": "bytes"},
        ],
        "name": "closeChannel",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "channelId", "type": "uint256"}],
        "name": "withdrawRemainder",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "channelCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    # Auto-generated public getter for `mapping(uint256 => Channel) channels`.
    # Status enum surfaces as uint8 (0 None, 1 Open, 2 Closing, 3 Settled).
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "channels",
        "outputs": [
            {"internalType": "address", "name": "client",   "type": "address"},
            {"internalType": "address", "name": "provider", "type": "address"},
            {"internalType": "uint256", "name": "deposit",  "type": "uint256"},
            {"internalType": "uint256", "name": "claimed",  "type": "uint256"},
            {"internalType": "uint256", "name": "closesAt", "type": "uint256"},
            {"internalType": "uint8",   "name": "status",   "type": "uint8"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "channelId", "type": "uint256"}],
        "name": "getChannel",
        "outputs": [
            {
                "components": [
                    {"internalType": "address", "name": "client",   "type": "address"},
                    {"internalType": "address", "name": "provider", "type": "address"},
                    {"internalType": "uint256", "name": "deposit",  "type": "uint256"},
                    {"internalType": "uint256", "name": "claimed",  "type": "uint256"},
                    {"internalType": "uint256", "name": "closesAt", "type": "uint256"},
                    {"internalType": "uint8",   "name": "status",   "type": "uint8"},
                ],
                "internalType": "struct PaymentChannel.Channel",
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "challengeWindow",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "uint256", "name": "channelId", "type": "uint256"},
            {"indexed": True,  "internalType": "address", "name": "client",    "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "provider",  "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "deposit",   "type": "uint256"},
        ],
        "name": "ChannelOpened",
        "type": "event",
    },
]


# Minimal ATN (Substrate ERC20 surface) ABI for approve() — the PaymentChannel
# pulls the deposit with substrate.transferFrom, so a client must approve the
# channel contract on Substrate before openChannel.
ATN_APPROVE_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "amount",  "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "owner",   "type": "address"},
            {"internalType": "address", "name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


# EIP-712 domain + Voucher typehash for the PaymentChannel. MUST match the
# contract EXACTLY: EIP712("AutonetPaymentChannel", "1") and
# keccak256("Voucher(uint256 channelId,uint256 cumulativeAmount)"). Used by the
# pure off-chain voucher digest builder (build_voucher_digest) so signing does
# NOT require a live-chain call to voucherHash().
PAYMENT_CHANNEL_EIP712_NAME = "AutonetPaymentChannel"
PAYMENT_CHANNEL_EIP712_VERSION = "1"


def build_voucher_digest(
    channel_id: int,
    cumulative_amount: int,
    channel_address: str,
    chain_id: int,
) -> bytes:
    """Build the EIP-712 digest a client signs for a PaymentChannel voucher.

    PURE / offline — no chain call. Reproduces
    ``PaymentChannel.voucherHash(channelId, cumulativeAmount)`` exactly:
    the typed struct ``Voucher(uint256 channelId,uint256 cumulativeAmount)``
    under the domain ``EIP712("AutonetPaymentChannel", "1")`` folded with
    ``chainId`` and the channel contract ``verifyingContract``. The returned
    32 bytes are the final digest the contract recovers against
    (``digest.recover(signature)``), so signing it raw yields a signature
    ``closeChannel`` accepts.
    """
    from eth_account.messages import encode_typed_data
    from web3 import Web3

    typed = {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "Voucher": [
                {"name": "channelId", "type": "uint256"},
                {"name": "cumulativeAmount", "type": "uint256"},
            ],
        },
        "primaryType": "Voucher",
        "domain": {
            "name": PAYMENT_CHANNEL_EIP712_NAME,
            "version": PAYMENT_CHANNEL_EIP712_VERSION,
            "chainId": int(chain_id),
            "verifyingContract": Web3.to_checksum_address(channel_address),
        },
        "message": {
            "channelId": int(channel_id),
            "cumulativeAmount": int(cumulative_amount),
        },
    }
    signable = encode_typed_data(full_message=typed)
    # The signable message body/header hash to the same 32-byte digest the
    # contract's _hashTypedDataV4 produces. Recompute it via eth_account's
    # own hasher so we hand back the exact digest recover() checks.
    from eth_account.messages import _hash_eip191_message
    return _hash_eip191_message(signable)


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

    async def register_agent_bound(
        self,
        identity: "AgentIdentity",
        private_key: str,
        peer_id: str | bytes,
        owner: str,
        owner_sig: str | bytes,
        parent: str = "",
    ) -> dict[str, Any]:
        """Sign and submit a ``registerAgentBound`` transaction.

        Mirrors :meth:`register_agent`, but binds the agent to a proven
        ``owner`` wallet (and optional ``parent`` agent). ``owner_sig`` is
        the owner wallet's EIP-712 ``OwnerBinding{agent, parent, nonce}``
        signature over the domain ("AutonetSubstrate", "1", chainId, this
        contract) at the agent's CURRENT ``bindingNonce`` — the contract
        verifies it on-chain (BadOwnerBindingSignature revert on mismatch).
        The tx is signed by the AGENT key (msg.sender), exactly like the
        legacy path. ``parent`` defaults to the zero address (top-level).
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)

            lineage_bytes = _to_bytes32(identity.lineage_hash)
            peer_bytes = _to_peer_id_bytes(peer_id)
            owner_addr = w3.to_checksum_address(owner)
            parent_addr = w3.to_checksum_address(parent) if parent else ZERO_ADDRESS
            sig_bytes = (owner_sig if isinstance(owner_sig, bytes)
                         else bytes.fromhex(owner_sig[2:] if owner_sig.startswith("0x")
                                            else owner_sig))

            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id

            try:
                estimated = contract.functions.registerAgentBound(
                    lineage_bytes, peer_bytes, owner_addr, parent_addr, sig_bytes,
                ).estimate_gas({"from": account.address})
                gas_limit = max(int(estimated * 12 // 10), 500_000)
            except Exception:
                gas_limit = 1_500_000

            tx = contract.functions.registerAgentBound(
                lineage_bytes, peer_bytes, owner_addr, parent_addr, sig_bytes,
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
                    "Agent registered (bound to %s) on substrate: %s tx=%s",
                    owner_addr, account.address, tx_hash.hex(),
                )
                return {
                    "success": True,
                    "tx_hash": tx_hash.hex(),
                    "agent_address": account.address,
                    "owner": owner_addr,
                    "parent": parent_addr,
                }
            return {
                "success": False,
                "error": "Transaction reverted",
                "tx_hash": tx_hash.hex(),
            }
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to register (bound) agent on chain")
            return {"success": False, "error": str(e)}

    async def rotate_owner(
        self,
        private_key: str,
        new_owner: str,
        new_owner_sig: str | bytes,
        new_parent: str = "",
    ) -> dict[str, Any]:
        """Sign and submit a ``rotateOwner`` transaction.

        Rotates (or first-sets) the calling agent's owner+parent to
        ``new_owner``. Authorized by CUSTODY of the agent key (the tx
        signer = msg.sender) plus the NEW owner's EIP-712 signature at the
        agent's current binding nonce. The OLD owner does NOT participate —
        this is the key-loss recovery path. ``new_parent`` defaults to the
        zero address (top-level).
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)

            owner_addr = w3.to_checksum_address(new_owner)
            parent_addr = (w3.to_checksum_address(new_parent)
                           if new_parent else ZERO_ADDRESS)
            sig_bytes = (new_owner_sig if isinstance(new_owner_sig, bytes)
                         else bytes.fromhex(new_owner_sig[2:]
                                            if new_owner_sig.startswith("0x")
                                            else new_owner_sig))

            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id

            try:
                estimated = contract.functions.rotateOwner(
                    owner_addr, parent_addr, sig_bytes,
                ).estimate_gas({"from": account.address})
                gas_limit = max(int(estimated * 12 // 10), 200_000)
            except Exception:
                gas_limit = 400_000

            tx = contract.functions.rotateOwner(
                owner_addr, parent_addr, sig_bytes,
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
                    "Agent %s owner rotated to %s tx=%s",
                    account.address, owner_addr, tx_hash.hex(),
                )
                return {
                    "success": True,
                    "tx_hash": tx_hash.hex(),
                    "agent_address": account.address,
                    "owner": owner_addr,
                    "parent": parent_addr,
                }
            return {
                "success": False,
                "error": "Transaction reverted",
                "tx_hash": tx_hash.hex(),
            }
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to rotate owner on chain")
            return {"success": False, "error": str(e)}

    def read_binding_status(self, address: str) -> dict[str, Any]:
        """Synchronous batched read of one agent's binding state.

        Three RPC round-trips (one when unregistered — the owner/nonce reads
        are skipped, an unregistered agent is unbound at nonce 0 by
        definition). Runs in a worker thread from the WS handler — NEVER on
        the event loop: a fleet of N agents would otherwise block the loop
        for N×RPC-latency and starve the websocket keepalive (observed live
        as 1011 disconnects with 15 agents). Raises on chain-read failure so
        the caller can surface it instead of reporting a misleading
        "unbound" state.
        """
        w3 = self._get_web3()
        contract = self._get_contract(w3)
        checksum = w3.to_checksum_address(address)
        registered = bool(contract.functions.isRegistered(checksum).call())
        if not registered:
            return {"registered": False, "owner": ZERO_ADDRESS, "nonce": 0}
        owner = contract.functions.agentOwner(checksum).call()
        nonce = int(contract.functions.bindingNonce(checksum).call())
        return {"registered": True, "owner": str(owner), "nonce": nonce}

    async def binding_nonce(self, address: str) -> int:
        """Read an agent's current owner-binding nonce (0 if never bound).

        This is the nonce the owner wallet must sign the next OwnerBinding
        against. Returns 0 on any read failure (a never-bound agent is at
        nonce 0 anyway).
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            return int(contract.functions.bindingNonce(
                w3.to_checksum_address(address)).call())
        except Exception:
            log.debug("bindingNonce read failed for %s", address, exc_info=True)
            return 0

    async def get_agent_owner(self, address: str) -> str:
        """Read an agent's bound owner wallet (zero address if unbound)."""
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            return contract.functions.agentOwner(
                w3.to_checksum_address(address)).call()
        except Exception:
            log.debug("agentOwner read failed for %s", address, exc_info=True)
            return ZERO_ADDRESS

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

    async def get_agent_endpoint(self, address: str) -> str:
        """Read an agent's current on-chain wss endpoint ('' if unset)."""
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            checksum = w3.to_checksum_address(address)
            return contract.functions.getAgentEndpoint(checksum).call() or ""
        except Exception:
            log.debug("getAgentEndpoint read failed for %s", address, exc_info=True)
            return ""

    async def update_endpoint(
        self,
        private_key: str,
        ws_endpoint: str,
    ) -> dict[str, Any]:
        """Sign and submit an ``updateEndpoint`` transaction (agent-signed
        presence). The indexer mirrors the EndpointUpdated event to the
        off-chain agent directory so browsers can resolve agent -> wss."""
        try:
            from eth_account import Account
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)
            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id
            tx = contract.functions.updateEndpoint(ws_endpoint).build_transaction({
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
            log.exception("Failed to update endpoint on chain")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Per-epoch training record (write)
    # ------------------------------------------------------------------

    async def record_training_for_epoch(
        self,
        private_key: str,
        amount: int,
        epoch_id_hash: str | bytes,
        proof: list[str | bytes],
    ) -> dict[str, Any]:
        """Sign and submit a ``recordTrainingForEpoch`` transaction.

        Money only (Decision 2026-07-10): ``amount`` is the ATN mint,
        federation-ratified — the merkle leaf under the anchor's
        agentMintRoot commits (agent, amount), so ``proof`` must have been
        built over the SAME amount (nodes/common/mint_merkle.py
        mint_merkle_proof) or the contract reverts MintProofInvalid. REP
        is no longer minted here; it is claimed DAO-side (RepToken) on
        ratified ATN earnings.

        Per-(agent, epoch) idempotent on-chain; the epoch must already be
        anchored via ``submitAnchor``.
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)

            eid = (epoch_id_hash if isinstance(epoch_id_hash, bytes)
                   else _to_bytes32(epoch_id_hash))
            proof_bytes = [
                (p if isinstance(p, bytes) else _to_bytes32(p)) for p in proof
            ]

            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id

            try:
                estimated = contract.functions.recordTrainingForEpoch(
                    int(amount), eid, proof_bytes,
                ).estimate_gas({"from": account.address})
                gas_limit = max(int(estimated * 12 // 10), 1_500_000)
            except Exception:
                gas_limit = 2_000_000

            tx = contract.functions.recordTrainingForEpoch(
                int(amount), eid, proof_bytes,
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
                return {"success": True, "tx_hash": tx_hash.hex()}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hash.hex()}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to record training for epoch on chain")
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
        """Read cumulative earnings + ATN balance + native gas balance.

        Money only (Decision 2026-07-10): Substrate has no reputation
        surface. The ``reputation`` field reports ``agentMintTotal`` —
        cumulative tool-pool earnings (money). True REP/voice is claimed
        DAO-side (RepToken) on these ratified earnings. ATN is the
        transferable ERC20-shaped balance; native gas comes from the chain.
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            addr = w3.to_checksum_address(address)
            reputation = contract.functions.agentMintTotal(addr).call()
            atn = contract.functions.balanceOf(addr).call()
            gas_balance = w3.eth.get_balance(addr)
            return {
                "address": addr,
                "reputation": str(reputation),
                "atn_balance": str(atn),
                "gas_balance": str(gas_balance),
            }
        except Exception as e:
            log.debug("Failed to read balances for %s: %s", address, e)
            return None

    async def get_provider_record(self, address: str) -> dict[str, Any] | None:
        """Read a service provider's VERIFIABLE track record.

        Services carry no reputation, no mint and no substrate standing —
        remote execution is unknowable in principle, so there is no honest
        quality score to show (docs/services_market.md). What IS honest is
        what the chain already proves about a counterparty:

          ``service_earnings``  cumulative NET ATN this address has been
                                paid for services (``serviceEarnings``,
                                bumped by ``payForService``; channel
                                settlements route through it too, so it
                                includes closed channels).
          ``agent_mint_total``  cumulative tool-pool earnings — the same
                                address authoring tools the network uses.
          ``registered_at``     identity tenure. An address that has been
                                serving for a year is a different
                                counterparty from one minted this morning.

        None of these can be self-reported: they are consequences of other
        people's payments. That is the whole point — this is a track
        record, not a rating.
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            addr = w3.to_checksum_address(address)
            earnings = contract.functions.serviceEarnings(addr).call()
            try:
                mint_total = contract.functions.agentMintTotal(addr).call()
            except Exception:                              # noqa: BLE001
                mint_total = 0
            registered_at = 0
            submissions = 0
            try:
                # (lineageHash, registeredAt, active, totalTrainingMint,
                #  trainingSubmissionCount) — see the `agents` ABI entry.
                agent = contract.functions.agents(addr).call()
                registered_at = int(agent[1])
                submissions = int(agent[4])
            except Exception:                              # noqa: BLE001
                pass
            return {
                "address": addr,
                "service_earnings": str(earnings),
                "agent_mint_total": str(mint_total),
                "registered_at": registered_at,
                "training_submissions": submissions,
            }
        except Exception as e:                             # noqa: BLE001
            log.debug("Failed to read provider record for %s: %s", address, e)
            return None

    async def get_fleet_voice(self, owner: str) -> dict[str, Any] | None:
        """Read an owner's fleet: their wallet ATN plus every bound
        agent's ATN + reputation, the network ATN supply (money) AND the
        network reputation supply (voice).

        VOICE is REPUTATION (ratified 2026-07-08: ATN = money, reputation
        = voice). ``fleet_reputation_raw`` (Σ bound agents' reputation) is
        the household's voice numerator — the number the federated close
        weights the fleet's reviews and usage by (see
        nodes/common/voice_state.py), over ``rep_supply_raw``. Owner
        wallets never earn reputation, so there is no owner-wallet
        reputation term.

        MONEY figures (owner balance, fleet ATN total, ATN supply) are
        still reported for the Owner page's money panel, but they no
        longer drive the voice weight. Raw integer chain units; the WS
        layer scales for display.
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            owner_addr = w3.to_checksum_address(owner)
            owner_raw = int(contract.functions.balanceOf(owner_addr).call())
            fleet_raw = owner_raw            # money: owner + Σ agent ATN
            fleet_rep_raw = 0                # voice: Σ agent reputation
            agents: list[dict[str, Any]] = []
            count = contract.functions.registeredAgentCount().call()
            for i in range(count):
                try:
                    addr = contract.functions.getRegisteredAgent(i).call()
                    bound = contract.functions.agentOwner(addr).call()
                    if str(bound).lower() != str(owner_addr).lower():
                        continue
                    bal = int(contract.functions.balanceOf(addr).call())
                    # Money only: agentMintTotal (cumulative pool earnings)
                    # stands in for the deleted agentReputation surface. True
                    # REP/voice is DAO-side (RepToken) on ratified earnings.
                    rep = int(contract.functions.agentMintTotal(addr).call())
                    agents.append({
                        "agent_id": str(addr),
                        "balance_raw": bal,
                        "reputation_raw": rep,
                    })
                    fleet_raw += bal
                    fleet_rep_raw += rep
                except Exception as e:
                    log.debug("fleet_voice: agent %d read failed: %s", i, e)
                    continue
            supply_raw = int(contract.functions.atnTotalSupply().call())
            rep_supply_raw = int(contract.functions.networkMintTotal().call())
            return {
                "owner": str(owner_addr),
                "owner_balance_raw": owner_raw,
                "fleet_total_raw": fleet_raw,
                "fleet_reputation_raw": fleet_rep_raw,
                "supply_raw": supply_raw,
                "rep_supply_raw": rep_supply_raw,
                "agents": agents,
            }
        except Exception as e:
            log.debug("Failed to read fleet voice for %s: %s", owner, e)
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

    # ------------------------------------------------------------------
    # Service payments (Substrate.payForService)
    # ------------------------------------------------------------------

    async def pay_for_service(
        self,
        private_key: str,
        recipient: str,
        amount: int,
        request_id: str | bytes,
    ) -> dict[str, Any]:
        """Sign and submit a ``Substrate.payForService`` transaction.

        A labeled ATN transfer from the AGENT key (msg.sender = the payer)
        to ``recipient``, taking the service fee and recording
        serviceEarnings on chain. ``amount`` is the GROSS ATN (the recipient
        receives net of the fee). ``request_id`` is the opaque off-chain
        request id (bytes32; e.g. keccak256 of the request body) emitted in
        the ServicePayment event so the payment is matchable off-chain.

        Signs with the agent key exactly like ``register_agent`` /
        ``record_training_for_epoch``, with Etherlink-tolerant explicit gas
        (estimate × 1.2, floored). Returns
        ``{success, tx_hash[, request_id]}`` or ``{success: False, error}``.
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)

            recipient_addr = w3.to_checksum_address(recipient)
            req = (request_id if isinstance(request_id, bytes)
                   else _to_bytes32(request_id))

            nonce = w3.eth.get_transaction_count(account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id

            try:
                estimated = contract.functions.payForService(
                    recipient_addr, int(amount), req,
                ).estimate_gas({"from": account.address})
                # ×1.6, floor 400k. NOT ×1.2: payForService writes up to three
                # ERC20Votes-style checkpoint pushes (payer, recipient,
                # treasury) plus a supply-history push, and on a FIRST payment
                # from a fresh wallet those are zero->nonzero SSTOREs the
                # estimator systematically undercounts. Measured on a local
                # hardhat node: estimate 217_524, actual 262_974 — a ×1.2
                # buffer lands just BELOW the true cost and the first payment
                # from every new agent wallet fails with "out of gas".
                gas_limit = max(int(estimated * 16 // 10), 400_000)
            except Exception:
                gas_limit = 600_000

            tx = contract.functions.payForService(
                recipient_addr, int(amount), req,
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
                return {
                    "success": True,
                    "tx_hash": tx_hash.hex(),
                    "request_id": req.hex() if isinstance(req, bytes) else str(req),
                }
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hash.hex()}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to pay for service on chain")
            return {"success": False, "error": str(e)}

    async def verify_service_payment(
        self,
        request_id: str | bytes,
        recipient: str,
        min_amount: int,
        payer: str | None = None,
        tx_hash: str | bytes | None = None,
    ) -> dict[str, Any]:
        """Verify a claimed on-chain service payment, cheaply.

        Etherlink caps log-range scans, so this NEVER scans a block range:
        a ``tx_hash`` is REQUIRED. It fetches that one receipt and decodes
        the ``ServicePayment`` event(s) from its logs (Substrate is the
        emitter), then checks the claimed facts:

          - ``requestId`` matches ``request_id``;
          - ``recipient`` matches ``recipient``;
          - ``amount`` (gross) is >= ``min_amount``;
          - if ``payer`` is given, the event's payer matches it.

        Returns a dict:
          ``{verified: bool, reason: str, payer, recipient, amount,
             request_id, tx_hash}``. ``verified`` is False (with a reason)
          on any mismatch, a missing/absent event, a reverted tx, or a read
          failure. ``amount`` is the gross ATN as recorded in the event.
        """
        result: dict[str, Any] = {
            "verified": False,
            "reason": "",
            "payer": None,
            "recipient": None,
            "amount": None,
            "request_id": None,
            "tx_hash": None,
        }
        if not tx_hash:
            result["reason"] = "tx_hash required (no log-range scan on Etherlink)"
            return result
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)

            recipient_addr = w3.to_checksum_address(recipient)
            want_req = (request_id if isinstance(request_id, bytes)
                        else _to_bytes32(request_id))
            th = tx_hash if isinstance(tx_hash, (bytes, str)) else str(tx_hash)
            result["tx_hash"] = th.hex() if isinstance(th, bytes) else str(th)

            receipt = w3.eth.get_transaction_receipt(th)
            # web3 returns an AttributeDict (attribute access); tests may pass a
            # plain dict. Read status via either shape.
            status = (receipt.get("status") if isinstance(receipt, dict)
                      else getattr(receipt, "status", 1))
            if status == 0:
                result["reason"] = "transaction reverted"
                return result

            # Decode ServicePayment events from THIS receipt only. process_receipt
            # with errors=DISCARD skips logs from other contracts / other events.
            try:
                from web3.logs import DISCARD
                events = contract.events.ServicePayment().process_receipt(
                    receipt, errors=DISCARD)
            except Exception:
                events = contract.events.ServicePayment().process_receipt(receipt)

            if not events:
                result["reason"] = "no ServicePayment event in receipt"
                return result

            # Find the event matching the claimed requestId + recipient.
            for ev in events:
                args = ev["args"]
                ev_req = args["requestId"]
                ev_req_b = (ev_req if isinstance(ev_req, bytes)
                            else _to_bytes32(str(ev_req)))
                ev_recipient = w3.to_checksum_address(args["recipient"])
                ev_payer = w3.to_checksum_address(args["payer"])
                ev_amount = int(args["amount"])

                if ev_req_b != want_req:
                    continue
                if ev_recipient != recipient_addr:
                    continue

                result.update({
                    "payer": ev_payer,
                    "recipient": ev_recipient,
                    "amount": ev_amount,
                    "request_id": ev_req_b.hex(),
                })
                if ev_amount < int(min_amount):
                    result["reason"] = (
                        f"amount {ev_amount} < min_amount {int(min_amount)}")
                    return result
                if payer is not None:
                    if ev_payer != w3.to_checksum_address(payer):
                        result["reason"] = "payer mismatch"
                        return result
                result["verified"] = True
                result["reason"] = "ok"
                return result

            result["reason"] = "no ServicePayment event matched requestId+recipient"
            return result
        except Exception as e:
            log.debug("verify_service_payment failed: %s", e, exc_info=True)
            result["reason"] = f"read failed: {e}"
            return result


# ---------------------------------------------------------------------------
# ServiceMarket client (ServiceRegistry + PaymentChannel)
# ---------------------------------------------------------------------------


class ServiceMarketClient:
    """Daemon-side helper for ServiceMarket.sol (ServiceRegistry +
    PaymentChannel).

    Follows the same shape as :class:`OnChainService`: constructed from the
    ``RPBConfig``, resolves contract addresses from config (never
    hardcoded), lazy-imports web3/eth-account, signs writes with the AGENT
    key (per-agent keypair passed in per call), and uses Etherlink-tolerant
    explicit gas (estimate × 1.2, floored). All writes return
    ``{success, tx_hash, ...}`` or ``{success: False, error}``; reads return
    the decoded value or ``None`` on failure.

    Addresses:
      - ServiceRegistry: ``config.service_registry_address`` (falls back to
        the legacy shared ``registry_address`` seed for back-compat).
      - PaymentChannel: ``config.payment_channel_address``.
      - ATN/Substrate (for the deposit approve): ``config.substrate_address``
        (falls back to ``rpb_contract_address``).
    """

    def __init__(self, config: "RPBConfig") -> None:
        self.config = config

    # ---- plumbing ----------------------------------------------------

    def _service_registry_address(self) -> str:
        addr = getattr(self.config, "service_registry_address", "") or ""
        if not addr:
            # Legacy seed lands the ServiceRegistry in registry_address too.
            addr = getattr(self.config, "registry_address", "") or ""
        return addr

    def _payment_channel_address(self) -> str:
        return getattr(self.config, "payment_channel_address", "") or ""

    def _substrate_address(self) -> str:
        addr = getattr(self.config, "substrate_address", "") or ""
        if not addr:
            addr = getattr(self.config, "rpb_contract_address", "") or ""
        return addr

    @property
    def registry_available(self) -> bool:
        return bool(self._service_registry_address() and self.config.rpc_url)

    @property
    def channel_available(self) -> bool:
        return bool(self._payment_channel_address() and self.config.rpc_url)

    def _get_web3(self):
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
        if not w3.is_connected():
            raise ConnectionError(f"Cannot connect to {self.config.rpc_url}")
        return w3

    def _registry_contract(self, w3=None):
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        addr = self._service_registry_address()
        if not addr:
            raise ValueError("service_registry_address not configured")
        return w3.eth.contract(
            address=Web3.to_checksum_address(addr),
            abi=SERVICE_REGISTRY_ABI,
        )

    def _channel_contract(self, w3=None):
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        addr = self._payment_channel_address()
        if not addr:
            raise ValueError("payment_channel_address not configured")
        return w3.eth.contract(
            address=Web3.to_checksum_address(addr),
            abi=PAYMENT_CHANNEL_ABI,
        )

    def _atn_contract(self, w3=None):
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        addr = self._substrate_address()
        if not addr:
            raise ValueError("substrate_address not configured")
        return w3.eth.contract(
            address=Web3.to_checksum_address(addr),
            abi=ATN_APPROVE_ABI,
        )

    @staticmethod
    def _send_signed(w3, account, func, chain_id: int, gas_floor: int):
        """Estimate gas (×1.2, floored), build/sign/send, wait for receipt.

        Shared write path — mirrors OnChainService's per-method inline
        pattern. Returns (receipt, tx_hash_hex).
        """
        nonce = w3.eth.get_transaction_count(account.address)
        try:
            estimated = func.estimate_gas({"from": account.address})
            gas_limit = max(int(estimated * 12 // 10), gas_floor)
        except Exception:
            gas_limit = gas_floor * 2
        tx = func.build_transaction({
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
        return receipt, tx_hash.hex()

    # ---- ServiceRegistry (writes) ------------------------------------

    async def register_service(
        self,
        private_key: str,
        spec_digest: str | bytes,
        ask_amount: int,
    ) -> dict[str, Any]:
        """``ServiceRegistry.registerService(specDigest, askAmount)``.

        Signed by the AGENT key (msg.sender must be a registered Substrate
        agent — the contract's ``onlyAgent`` gate). ``spec_digest`` is the
        sha256 of the service_spec blob (bytes32; NOT an IPFS CID).
        ``ask_amount`` is the per-work-item price in ATN units.

        The serviceId is assigned on-chain (sequential); it is decoded from
        the ServiceRegistered event in the receipt and returned as
        ``service_id`` (None if the event couldn't be decoded).
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._registry_contract(w3)
            account = Account.from_key(private_key)
            chain_id = self.config.chain_id or w3.eth.chain_id

            digest = (spec_digest if isinstance(spec_digest, bytes)
                      else _to_bytes32(spec_digest))
            func = contract.functions.registerService(digest, int(ask_amount))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 300_000)
            if receipt.status != 1:
                return {"success": False, "error": "Transaction reverted",
                        "tx_hash": tx_hex}
            service_id = None
            try:
                from web3.logs import DISCARD
                evs = contract.events.ServiceRegistered().process_receipt(
                    receipt, errors=DISCARD)
                if evs:
                    service_id = int(evs[0]["args"]["serviceId"])
            except Exception:
                log.debug("register_service: event decode failed", exc_info=True)
            return {"success": True, "tx_hash": tx_hex, "service_id": service_id}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to register service on chain")
            return {"success": False, "error": str(e)}

    async def update_service_ask(
        self,
        private_key: str,
        service_id: int,
        ask_amount: int,
    ) -> dict[str, Any]:
        """``ServiceRegistry.updateServiceAsk(serviceId, askAmount)`` (provider-only)."""
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._registry_contract(w3)
            account = Account.from_key(private_key)
            chain_id = self.config.chain_id or w3.eth.chain_id
            func = contract.functions.updateServiceAsk(
                int(service_id), int(ask_amount))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 120_000)
            if receipt.status == 1:
                return {"success": True, "tx_hash": tx_hex}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hex}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to update service ask on chain")
            return {"success": False, "error": str(e)}

    async def retire_service(
        self,
        private_key: str,
        service_id: int,
    ) -> dict[str, Any]:
        """``ServiceRegistry.retireService(serviceId)`` (provider-only)."""
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._registry_contract(w3)
            account = Account.from_key(private_key)
            chain_id = self.config.chain_id or w3.eth.chain_id
            func = contract.functions.retireService(int(service_id))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 120_000)
            if receipt.status == 1:
                return {"success": True, "tx_hash": tx_hex}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hex}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to retire service on chain")
            return {"success": False, "error": str(e)}

    # ---- ServiceRegistry (reads) -------------------------------------

    async def service_count(self) -> int:
        """``ServiceRegistry.serviceCount()`` (0 on read failure)."""
        try:
            w3 = self._get_web3()
            contract = self._registry_contract(w3)
            return int(contract.functions.serviceCount().call())
        except Exception:
            log.debug("service_count read failed", exc_info=True)
            return 0

    async def get_service(self, service_id: int) -> dict[str, Any] | None:
        """``ServiceRegistry.getService(serviceId)`` → decoded dict.

        Returns None if the service doesn't exist (provider == zero
        address) or on a read failure. Fields:
        ``{service_id, provider, spec_digest, ask_amount, active,
        registered_at, updated_at}``.
        """
        try:
            w3 = self._get_web3()
            contract = self._registry_contract(w3)
            s = contract.functions.getService(int(service_id)).call()
            provider, spec_digest, ask_amount, active, registered_at, updated_at = s
            if int(registered_at) == 0 or str(provider) == ZERO_ADDRESS:
                return None
            return {
                "service_id": int(service_id),
                "provider": str(provider),
                "spec_digest": spec_digest.hex() if isinstance(spec_digest, bytes) else str(spec_digest),
                "ask_amount": str(ask_amount),
                "active": bool(active),
                "registered_at": int(registered_at),
                "updated_at": int(updated_at),
            }
        except Exception as e:
            log.debug("get_service(%s) failed: %s", service_id, e)
            return None

    async def list_services(
        self, start: int = 1, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Enumerate services from ``start`` (serviceId is 1-indexed; index 0
        is unused). Reads ``serviceCount`` then ``getService(i)`` in order,
        skipping any that fail to decode. ``limit`` caps the count returned.
        """
        try:
            count = await self.service_count()
            out: list[dict[str, Any]] = []
            last = count if limit is None else min(count, start - 1 + limit)
            for i in range(max(start, 1), last + 1):
                rec = await self.get_service(i)
                if rec is not None:
                    out.append(rec)
            return out
        except Exception as e:
            log.debug("list_services failed: %s", e)
            return []

    # ---- PaymentChannel (deposit approve + writes) -------------------

    async def approve_channel_deposit(
        self,
        private_key: str,
        amount: int,
    ) -> dict[str, Any]:
        """Approve the PaymentChannel to pull ``amount`` ATN via
        ``Substrate.approve(channel, amount)``.

        openChannel pulls the deposit with ``substrate.transferFrom``, so
        the client MUST approve the channel contract on Substrate first.
        Signed by the client (AGENT) key.
        """
        try:
            from eth_account import Account
            from web3 import Web3

            w3 = self._get_web3()
            atn = self._atn_contract(w3)
            account = Account.from_key(private_key)
            chain_id = self.config.chain_id or w3.eth.chain_id
            channel_addr = Web3.to_checksum_address(self._payment_channel_address())
            func = atn.functions.approve(channel_addr, int(amount))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 120_000)
            if receipt.status == 1:
                return {"success": True, "tx_hash": tx_hex}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hex}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to approve channel deposit on chain")
            return {"success": False, "error": str(e)}

    async def open_channel(
        self,
        private_key: str,
        provider: str,
        deposit: int,
        auto_approve: bool = True,
    ) -> dict[str, Any]:
        """``PaymentChannel.openChannel(provider, deposit)``.

        The client escrows ``deposit`` ATN to ``provider``. openChannel pulls
        the deposit via ``substrate.transferFrom``, so an allowance must
        exist first: when ``auto_approve`` (default), this checks the current
        allowance and issues an ``approve`` for the shortfall before opening.
        Signed by the client (AGENT) key.

        The channelId is assigned on-chain (sequential); it is decoded from
        the ChannelOpened event and returned as ``channel_id`` (None if the
        event couldn't be decoded). Also returns ``approve_tx_hash`` when an
        approve was issued.
        """
        try:
            from eth_account import Account
            from web3 import Web3

            w3 = self._get_web3()
            channel = self._channel_contract(w3)
            account = Account.from_key(private_key)
            chain_id = self.config.chain_id or w3.eth.chain_id
            provider_addr = w3.to_checksum_address(provider)

            approve_tx_hash = None
            if auto_approve:
                atn = self._atn_contract(w3)
                channel_addr = Web3.to_checksum_address(
                    self._payment_channel_address())
                try:
                    current = int(atn.functions.allowance(
                        account.address, channel_addr).call())
                except Exception:
                    current = 0
                if current < int(deposit):
                    ap = await self.approve_channel_deposit(
                        private_key, int(deposit))
                    if not ap.get("success"):
                        return {"success": False,
                                "error": f"approve failed: {ap.get('error')}"}
                    approve_tx_hash = ap.get("tx_hash")

            func = channel.functions.openChannel(provider_addr, int(deposit))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 300_000)
            if receipt.status != 1:
                return {"success": False, "error": "Transaction reverted",
                        "tx_hash": tx_hex, "approve_tx_hash": approve_tx_hash}
            channel_id = None
            try:
                from web3.logs import DISCARD
                evs = channel.events.ChannelOpened().process_receipt(
                    receipt, errors=DISCARD)
                if evs:
                    channel_id = int(evs[0]["args"]["channelId"])
            except Exception:
                log.debug("open_channel: event decode failed", exc_info=True)
            return {"success": True, "tx_hash": tx_hex,
                    "channel_id": channel_id,
                    "approve_tx_hash": approve_tx_hash}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to open channel on chain")
            return {"success": False, "error": str(e)}

    def sign_voucher(
        self,
        private_key: str,
        channel_id: int,
        cumulative_amount: int,
    ) -> str:
        """Sign a PaymentChannel voucher over (channelId, cumulativeAmount).

        PURE / offline — builds the EIP-712 digest with
        :func:`build_voucher_digest` (matching the contract's
        ``EIP712("AutonetPaymentChannel","1")`` domain + ``Voucher(uint256
        channelId,uint256 cumulativeAmount)`` typehash) using the configured
        channel address + chain id, then signs the raw 32-byte digest with
        ``private_key`` (the CLIENT key). ``cumulativeAmount`` is the GROSS,
        MONOTONE running total the client authorizes to date.

        Returns the 0x-prefixed hex signature ``closeChannel`` accepts.
        """
        from eth_account import Account

        channel_addr = self._payment_channel_address()
        if not channel_addr:
            raise ValueError("payment_channel_address not configured")
        chain_id = int(self.config.chain_id or 0)
        if not chain_id:
            # Fall back to a live read only if config has no chain_id.
            chain_id = self._get_web3().eth.chain_id
        digest = build_voucher_digest(
            int(channel_id), int(cumulative_amount), channel_addr, chain_id)
        acct = Account.from_key(private_key)
        signed = (acct.unsafe_sign_hash(digest)
                  if hasattr(acct, "unsafe_sign_hash")
                  else Account._sign_hash(digest, acct._private_key))
        sig = signed.signature
        return sig.hex() if not sig.hex().startswith("0x") else sig.hex()

    async def verify_voucher(
        self,
        channel_id: int,
        cumulative_amount: int,
        signature: str | bytes,
    ) -> dict[str, Any]:
        """Recover a voucher's signer and check it against on-chain channel state.

        Rebuilds the EIP-712 digest offline (:func:`build_voucher_digest`),
        recovers the signer from ``signature``, then reads the channel and
        verifies: the channel exists and is Open (status == 1), the recovered
        signer equals the channel's ``client``, and (advisory) reports whether
        ``cumulative_amount`` exceeds the deposit (the contract caps the payout
        at the deposit, so an over-deposit voucher is not a signature failure).

        Returns ``{valid, reason, signer, client, provider, deposit, status,
        exceeds_deposit}``. ``valid`` is True only when the signer matches the
        channel client AND the channel is Open.
        """
        result: dict[str, Any] = {
            "valid": False, "reason": "", "signer": None, "client": None,
            "provider": None, "deposit": None, "status": None,
            "exceeds_deposit": None,
        }
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct  # noqa: F401
            from web3 import Web3

            channel_addr = self._payment_channel_address()
            if not channel_addr:
                result["reason"] = "payment_channel_address not configured"
                return result
            chain_id = int(self.config.chain_id or 0)
            w3 = self._get_web3()
            if not chain_id:
                chain_id = w3.eth.chain_id
            digest = build_voucher_digest(
                int(channel_id), int(cumulative_amount), channel_addr, chain_id)
            sig = (signature if isinstance(signature, bytes)
                   else bytes.fromhex(signature[2:] if signature.startswith("0x")
                                      else signature))
            # Recover the signer of the raw typed-data digest.
            signer = (Account._recover_hash(digest, signature=sig)
                      if hasattr(Account, "_recover_hash")
                      else Account.recoverHash(digest, signature=sig))
            signer = Web3.to_checksum_address(signer)
            result["signer"] = signer

            channel = self._channel_contract(w3)
            ch = channel.functions.getChannel(int(channel_id)).call()
            client, provider, deposit, claimed, closes_at, status = ch
            result.update({
                "client": Web3.to_checksum_address(client),
                "provider": Web3.to_checksum_address(provider),
                "deposit": str(deposit),
                "status": int(status),
                "exceeds_deposit": int(cumulative_amount) > int(deposit),
            })
            if int(status) != 1:  # 1 == Status.Open
                result["reason"] = f"channel not Open (status={int(status)})"
                return result
            if signer != Web3.to_checksum_address(client):
                result["reason"] = "signer is not the channel client"
                return result
            result["valid"] = True
            result["reason"] = "ok"
            return result
        except Exception as e:
            log.debug("verify_voucher failed: %s", e, exc_info=True)
            result["reason"] = f"verify failed: {e}"
            return result

    async def close_channel(
        self,
        private_key: str,
        channel_id: int,
        cumulative_amount: int,
        signature: str | bytes,
    ) -> dict[str, Any]:
        """``PaymentChannel.closeChannel(channelId, cumulativeAmount, signature)``.

        Submitted by the PROVIDER (msg.sender must be the channel provider).
        Pays the provider min(cumulative, deposit) NET of the service fee and
        opens the challenge window. ``signature`` is the client's voucher
        signature over (channelId, cumulativeAmount).
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            channel = self._channel_contract(w3)
            account = Account.from_key(private_key)
            chain_id = self.config.chain_id or w3.eth.chain_id
            sig = (signature if isinstance(signature, bytes)
                   else bytes.fromhex(signature[2:] if signature.startswith("0x")
                                      else signature))
            func = channel.functions.closeChannel(
                int(channel_id), int(cumulative_amount), sig)
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 200_000)
            if receipt.status == 1:
                return {"success": True, "tx_hash": tx_hex}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hex}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to close channel on chain")
            return {"success": False, "error": str(e)}

    async def withdraw_remainder(
        self,
        private_key: str,
        channel_id: int,
    ) -> dict[str, Any]:
        """``PaymentChannel.withdrawRemainder(channelId)``.

        Permissionless (it's a timer): after the challenge window elapses,
        refunds the client's remainder. Anyone may trigger it; funds only
        ever go to the client. Signed by whatever ``private_key`` is passed.
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            channel = self._channel_contract(w3)
            account = Account.from_key(private_key)
            chain_id = self.config.chain_id or w3.eth.chain_id
            func = channel.functions.withdrawRemainder(int(channel_id))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 120_000)
            if receipt.status == 1:
                return {"success": True, "tx_hash": tx_hex}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hex}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to withdraw remainder on chain")
            return {"success": False, "error": str(e)}

    async def get_channel(self, channel_id: int) -> dict[str, Any] | None:
        """``PaymentChannel.getChannel(channelId)`` → decoded dict.

        Returns None if the channel doesn't exist (status == 0 / None) or on
        a read failure. Fields: ``{channel_id, client, provider, deposit,
        claimed, closes_at, status}`` (status: 0 None, 1 Open, 2 Closing,
        3 Settled).
        """
        try:
            w3 = self._get_web3()
            channel = self._channel_contract(w3)
            ch = channel.functions.getChannel(int(channel_id)).call()
            client, provider, deposit, claimed, closes_at, status = ch
            if int(status) == 0:
                return None
            return {
                "channel_id": int(channel_id),
                "client": str(client),
                "provider": str(provider),
                "deposit": str(deposit),
                "claimed": str(claimed),
                "closes_at": int(closes_at),
                "status": int(status),
            }
        except Exception as e:
            log.debug("get_channel(%s) failed: %s", channel_id, e)
            return None


# ---------------------------------------------------------------------------
# CharterAnchor (governed charter-version anchor)
# ---------------------------------------------------------------------------

CHARTER_ANCHOR_ABI = [
    {
        "inputs": [],
        "name": "currentCharter",
        "outputs": [
            {"internalType": "uint256", "name": "version", "type": "uint256"},
            {"internalType": "bytes32", "name": "hash", "type": "bytes32"},
            {"internalType": "string", "name": "uri", "type": "string"},
            {"internalType": "bytes32", "name": "prevHash", "type": "bytes32"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "versionCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _read_current_charter_hash(w3, anchor_address: str) -> tuple[int, str] | None:
    """Read (version, charterHash-hex) from CharterAnchor.currentCharter().

    Returns None if nothing is anchored yet (the view reverts with
    ``NoCharter``) or the read fails. ``w3`` is a connected Web3 instance (or
    any object exposing the same ``eth.contract(...).functions.*.call()``
    surface — the tests pass a fake reader).
    """
    from web3 import Web3

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(anchor_address),
        abi=CHARTER_ANCHOR_ABI,
    )
    version, chain_hash, _uri, _prev, _ts = contract.functions.currentCharter().call()
    hexhash = chain_hash.hex() if isinstance(chain_hash, bytes) else str(chain_hash)
    if hexhash.startswith("0x"):
        hexhash = hexhash[2:]
    return int(version), hexhash.lower()


def verify_charter_against_anchor(w3, anchor_address: str) -> dict[str, Any]:
    """Compare the local charter_hash to the on-chain anchored charter.

    Reads ``CharterAnchor.currentCharter()`` and diffs its charterHash against
    the daemon's locally-computed :func:`charter_hash`. On mismatch the daemon
    logs a LOUD warning — this is a forward-only fork boundary and a divergent
    daemon would produce a different close (following the anchored charter is
    future migration work; detection comes first).

    Returns a dict: ``{match, local_hash, chain_hash, chain_version}``.
    ``match`` is None when the anchor read failed or nothing is anchored yet.
    """
    from nodes.common.world_model_substrate.charter_version import charter_hash

    local = charter_hash()
    try:
        read = _read_current_charter_hash(w3, anchor_address)
    except Exception as e:  # includes the NoCharter revert
        log.debug("Charter anchor read failed (%s); cannot verify.", e)
        return {"match": None, "local_hash": local, "chain_hash": None,
                "chain_version": None}

    if read is None:
        return {"match": None, "local_hash": local, "chain_hash": None,
                "chain_version": None}

    chain_version, chain_hash = read
    match = (chain_hash == local)
    if not match:
        log.warning(
            "CHARTER DIVERGENCE: local charter_hash=%s does NOT match anchored "
            "charter v%d hash=%s (anchor=%s). This daemon is running a different "
            "charter than the jurisdiction's governor anchored — its closes will "
            "not be bit-identical to the canonical charter. Follow-the-anchor "
            "migration is future work; this is detection only.",
            local, chain_version, chain_hash, anchor_address,
        )
    else:
        log.info("Charter matches anchored version v%d (hash=%s).",
                 chain_version, chain_hash)
    return {"match": match, "local_hash": local, "chain_hash": chain_hash,
            "chain_version": chain_version}
