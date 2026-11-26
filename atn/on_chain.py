"""On-chain agent registration + reads against ``Substrate.sol``.

This module is the daemon's interface to the substrate-native chain
surface. It supersedes the pre-substrate ``RPB`` contract integration
(which targeted a contract that no longer exists post-Phase 5.6a nuke).

What lives here
---------------

- ``register_agent_bound`` / ``sign_binding`` — register an agent,
  owner-bound (or parent-signed), with a libp2p PeerId so daemons can
  DHT-resolve it later. ``revoke_agent`` relays the owner's revocation.
- ``update_peer_id`` / ``build_update_peer_id_call_data`` — rotate
  the libp2p keypair without losing on-chain identity.
- ``is_registered``, ``get_agent_record`` — basic membership reads.
- ``get_agent_peer_id`` — for off-chain peer discovery.
- ``get_all_registered_agents`` — enumerate agents (for discovery).
- ``index_tokens`` / ``accepts_token``: the value index's tokens (what
  services are priced and paid in).
- ``get_agent_balances``: an address's balance of every index token, its
  gas balance and the value it has claimed from the pools.
- ``pay_for_service`` / ``verify_service_payment``: a direct payment in an
  index token, and the provider's check of one.
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

import asyncio
import functools
import hashlib
import inspect
import logging
import threading
from typing import Any, TYPE_CHECKING
from nodes.common.tx_nonce import next_nonce

if TYPE_CHECKING:
    from .config import RPBConfig
    from .models import AgentIdentity

log = logging.getLogger(__name__)


# ABI for Substrate.sol — only the surface we actually call. Kept inline
# so on_chain.py doesn't need the artifact JSON to load.
SUBSTRATE_ABI = [
    # Agent registration / discovery
    {
        "inputs": [{"internalType": "bytes", "name": "peerId", "type": "bytes"}],
        "name": "updatePeerId",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    # Owner-rooted registration, the only registration path (docs/
    # multi_jurisdiction.md decision 7). The AGENT (msg.sender) submits; the
    # OWNER wallet, or the live parent agent, has signed an EIP-712
    # OwnerBinding{agent, parent, nonce} over the domain ("AutonetSubstrate",
    # "1", chainId, this contract). rotateOwner needs NO old-owner sig
    # (key-loss recovery); the per-agent bindingNonce closes replay.
    {
        "inputs": [
            {"internalType": "bytes",   "name": "peerId",       "type": "bytes"},
            {"internalType": "address", "name": "owner",        "type": "address"},
            {"internalType": "address", "name": "parentAgent",  "type": "address"},
            {"internalType": "bytes",   "name": "sig",          "type": "bytes"},
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
        "inputs": [
            {"internalType": "address", "name": "agent",    "type": "address"},
            {"internalType": "bytes",   "name": "ownerSig", "type": "bytes"},
        ],
        "name": "revokeAgent",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "agent", "type": "address"}],
        "name": "isLive",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
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
            {"internalType": "uint256", "name": "registeredAt", "type": "uint256"},
            {"internalType": "bool",    "name": "active", "type": "bool"},
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
    # What an earner has claimed from the pools, as value (6-decimal value
    # units, at the parities stored with each anchor): what the DAO's REP
    # claim reads.
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "earnedValue",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalEarnedValue",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    # The part of earnedValue the REP floor seeded (value units).
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "seededValue",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    # The value index (Registry jurisdiction.parity.<token>) as Substrate
    # reads it, and the pools' tokens.
    {
        "inputs": [],
        "name": "registry",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "token", "type": "address"}],
        "name": "acceptsToken",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "token", "type": "address"}],
        "name": "parityOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "token",  "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "valueOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "poolTokens",
        "outputs": [{"internalType": "address[]", "name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "toolShareTotal",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "claimedTotal",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "token", "type": "address"}],
        "name": "poolCap",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    # Household claims (Decision 2026-10-27). The leaf is
    # keccak256(bytes.concat(keccak256(abi.encode(CLAIM_TAG, earner, tokens,
    # amounts, seed)))) under the anchor's claimRoot
    # (nodes/common/mint_merkle.py). Permissionless: anyone submits; the
    # claim pays ``earner`` (the household's owner wallet) the whole basket
    # in the same transaction. A token whose transfer fails stays owed
    # (PayDeferred) and leaves later by withdraw, one token at a time.
    # Substrate records the REP value on chain, the seed (the REP floor's
    # part, value units) included.
    {
        "inputs": [
            {"internalType": "address",   "name": "earner",      "type": "address"},
            {"internalType": "bytes32",   "name": "epochIdHash", "type": "bytes32"},
            {"internalType": "address[]", "name": "tokens",      "type": "address[]"},
            {"internalType": "uint256[]", "name": "amounts",     "type": "uint256[]"},
            {"internalType": "uint256",   "name": "seed",        "type": "uint256"},
            {"internalType": "bytes32[]", "name": "proof",       "type": "bytes32[]"},
        ],
        "name": "claim",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "earner", "type": "address"},
            {"internalType": "address", "name": "token",  "type": "address"},
        ],
        "name": "withdraw",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "", "type": "address"},
            {"internalType": "address", "name": "", "type": "address"},
        ],
        "name": "owed",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "", "type": "address"},
            {"internalType": "bytes32", "name": "", "type": "bytes32"},
        ],
        "name": "claimedEpoch",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    # Service payments in an index token: the payer approves Substrate for
    # the gross on the token; Substrate takes the fee and moves the net to
    # the recipient. Pricing is off chain (the provider's signed list).
    {
        "inputs": [
            {"internalType": "address", "name": "token",     "type": "address"},
            {"internalType": "address", "name": "recipient", "type": "address"},
            {"internalType": "uint256", "name": "gross",     "type": "uint256"},
            {"internalType": "bytes32", "name": "requestId", "type": "bytes32"},
        ],
        "name": "payForService",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "gross", "type": "uint256"}],
        "name": "feeOf",
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
    # ServicePayment / ServiceFee events: decoded from a tx receipt by
    # verify_service_payment (no log-range scan; Etherlink caps ranges),
    # and scanned by recipient for a provider's track record.
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "payer",     "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "recipient", "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "token",     "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount",    "type": "uint256"},
            {"indexed": False, "internalType": "bytes32", "name": "requestId", "type": "bytes32"},
        ],
        "name": "ServicePayment",
        "type": "event",
    },
    # One per token a claim credits: summed per earner for the basket so far.
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "earner",      "type": "address"},
            {"indexed": True,  "internalType": "bytes32", "name": "epochIdHash", "type": "bytes32"},
            {"indexed": True,  "internalType": "address", "name": "token",       "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount",      "type": "uint256"},
        ],
        "name": "ClaimedToken",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "payer",      "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "recipient",  "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "token",      "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount",     "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "toolShare",  "type": "uint256"},
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
    # Events used for log replay
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "address", "name": "agent",       "type": "address"},
            {"indexed": False, "internalType": "bytes",   "name": "peerId",      "type": "bytes"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp",   "type": "uint256"},
        ],
        "name": "AgentRegistered",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "agent", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "owner", "type": "address"},
        ],
        "name": "AgentRevoked",
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
            {"internalType": "address", "name": "token",    "type": "address"},
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
            {"internalType": "address", "name": "token",            "type": "address"},
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
            {"internalType": "uint256", "name": "upTo",             "type": "uint256"},
        ],
        "name": "closeChannel",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "channelId",        "type": "uint256"},
            {"internalType": "uint256", "name": "cumulativeAmount", "type": "uint256"},
            {"internalType": "bytes",   "name": "signature",        "type": "bytes"},
            {"internalType": "uint256", "name": "upTo",             "type": "uint256"},
        ],
        "name": "settle",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "channelId", "type": "uint256"},
            {"internalType": "uint256", "name": "amount",    "type": "uint256"},
        ],
        "name": "topUp",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "channelId", "type": "uint256"}],
        "name": "startClose",
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
    # Status enum surfaces as uint8 (0 None, 1 Open, 2 Closing, 3 Settled,
    # 4 ClientClosing).
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "channels",
        "outputs": [
            {"internalType": "address", "name": "client",   "type": "address"},
            {"internalType": "address", "name": "provider", "type": "address"},
            {"internalType": "address", "name": "token",    "type": "address"},
            {"internalType": "uint256", "name": "deposit",  "type": "uint256"},
            {"internalType": "uint256", "name": "claimed",  "type": "uint256"},
            {"internalType": "uint256", "name": "closesAt", "type": "uint256"},
            {"internalType": "uint8",   "name": "status",   "type": "uint8"},
            {"internalType": "uint256", "name": "window",   "type": "uint256"},
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
                    {"internalType": "address", "name": "token",    "type": "address"},
                    {"internalType": "uint256", "name": "deposit",  "type": "uint256"},
                    {"internalType": "uint256", "name": "claimed",  "type": "uint256"},
                    {"internalType": "uint256", "name": "closesAt", "type": "uint256"},
                    {"internalType": "uint8",   "name": "status",   "type": "uint8"},
                    {"internalType": "uint256", "name": "window",   "type": "uint256"},
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
            {"indexed": False, "internalType": "address", "name": "token",     "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "deposit",   "type": "uint256"},
        ],
        "name": "ChannelOpened",
        "type": "event",
    },
    # Every payout to the provider (settle, and a close that pays
    # something): amount paid now (gross) and the cumulative paid so far.
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "uint256", "name": "channelId", "type": "uint256"},
            {"indexed": True,  "internalType": "address", "name": "client",    "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "provider",  "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount",    "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "paidTotal", "type": "uint256"},
        ],
        "name": "ChannelPaid",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "uint256", "name": "channelId", "type": "uint256"},
            {"indexed": True,  "internalType": "address", "name": "client",    "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "amount",    "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "deposit",   "type": "uint256"},
        ],
        "name": "ChannelToppedUp",
        "type": "event",
    },
]

# eth_getLogs spans are read in chunks of this many blocks (public RPCs cap
# the range), as the service catalog does.
CHANNEL_LOG_CHUNK = 1000

# Per (rpc, Substrate, earner): the next block to scan for ClaimedToken logs
# and the totals summed so far (OnChainService._claimed_totals).
_CLAIMED_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_CLAIMED_LOCK = threading.Lock()


# The ERC-20 surface of an index token: a payer approves Substrate (a direct
# payment) or the PaymentChannel (a deposit) on the token itself before
# paying, and balances and decimals are the token's own. A token whose
# approve returns nothing (USDT) works too: a transaction's return value is
# never decoded.
ERC20_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "to",     "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
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

# The DAO Registry's key enumeration: the value index is every non-zero
# ``jurisdiction.parity.<token>`` (0x and 40 lowercase hex digits).
REGISTRY_ENUM_ABI = [
    {"name": "getAllKeys", "type": "function", "stateMutability": "view",
     "inputs": [], "outputs": [{"name": "", "type": "string[]"}]},
]
PARITY_KEY_PREFIX = "jurisdiction.parity."

# Gas floors for a direct payment, under the estimate x1.3. Measured on
# eth_tester (Cancun): the first payForService in a token (it lists the
# token, reads its parity from the Registry and opens its fee and
# tool-share checkpoints) uses about 396k (6 decimals) or 371k (18), a
# later one about 173k; an approve about 46k. The floor covers a first
# payment even when the estimate undercounts it.
PAY_GAS_FLOOR = 500_000
APPROVE_GAS_FLOOR = 80_000


# EIP-712 domain + Voucher typehash for the PaymentChannel. MUST match the
# contract EXACTLY: EIP712("AutonetPaymentChannel", "1") and
# keccak256("Voucher(uint256 channelId,address token,uint256 cumulativeAmount)").
# Used by the pure off-chain voucher digest builder (build_voucher_digest) so
# signing does NOT require a live-chain call to voucherHash().
PAYMENT_CHANNEL_EIP712_NAME = "AutonetPaymentChannel"
PAYMENT_CHANNEL_EIP712_VERSION = "1"


def build_voucher_digest(
    channel_id: int,
    token: str,
    cumulative_amount: int,
    channel_address: str,
    chain_id: int,
) -> bytes:
    """Build the EIP-712 digest a client signs for a PaymentChannel voucher.

    PURE / offline: no chain call. Reproduces
    ``PaymentChannel.voucherHash(channelId, token, cumulativeAmount)``
    exactly: the typed struct ``Voucher(uint256 channelId,address token,
    uint256 cumulativeAmount)`` under the domain
    ``EIP712("AutonetPaymentChannel", "1")`` folded with ``chainId`` and the
    channel contract ``verifyingContract``. ``token`` is the channel's
    token, so a voucher signed for one token never pays from a channel in
    another. The returned 32 bytes are the final digest the contract
    recovers against (``digest.recover(signature)``), so signing it raw
    yields a signature ``settle`` and ``closeChannel`` accept.
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
                {"name": "token", "type": "address"},
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
            "token": Web3.to_checksum_address(token),
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


_off_loop_state = threading.local()


def _off_the_loop(cls):
    """Run every ``async def`` of ``cls`` on a worker thread.

    The methods are async in name only: their bodies make blocking web3
    calls (RPC reads, and writes that wait for a receipt). Awaited on the
    daemon's event loop, each one froze the loop for the whole round trip;
    behind a slow egress proxy that stalled the WS server and gossip ingest
    for minutes (issues #40, #49). Each call now runs on its own loop in a
    worker thread; a method calling a sibling from there runs inline."""

    def wrap(fn):
        @functools.wraps(fn)
        async def run(*args, **kwargs):
            if getattr(_off_loop_state, "active", False):
                return await fn(*args, **kwargs)

            def worker():
                _off_loop_state.active = True
                try:
                    return asyncio.run(fn(*args, **kwargs))
                finally:
                    _off_loop_state.active = False

            return await asyncio.to_thread(worker)
        return run

    for name, fn in list(vars(cls).items()):
        if inspect.iscoroutinefunction(fn):
            setattr(cls, name, wrap(fn))
    return cls


@_off_the_loop
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

    async def register_agent_bound(
        self,
        identity: "AgentIdentity",
        private_key: str,
        peer_id: str | bytes,
        owner: str,
        owner_sig: str | bytes,
        parent: str = "",
    ) -> dict[str, Any]:
        """Sign and submit a ``registerAgentBound`` transaction, the only
        way an agent registers.

        Binds the agent to a proven ``owner`` wallet (and optional ``parent``
        agent). ``owner_sig`` is the EIP-712 ``OwnerBinding{agent, parent,
        nonce}`` signature over the domain ("AutonetSubstrate", "1", chainId,
        this contract) at the agent's CURRENT ``bindingNonce``, signed by the
        owner wallet or, when ``parent`` is set, by the parent agent's key
        (see :meth:`sign_binding`). The contract verifies it on-chain
        (BadOwnerBindingSignature revert on mismatch). The tx is signed by
        the AGENT key (msg.sender). ``parent`` defaults to the zero address
        (top-level).
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)

            peer_bytes = _to_peer_id_bytes(peer_id)
            owner_addr = w3.to_checksum_address(owner)
            parent_addr = w3.to_checksum_address(parent) if parent else ZERO_ADDRESS
            sig_bytes = (owner_sig if isinstance(owner_sig, bytes)
                         else bytes.fromhex(owner_sig[2:] if owner_sig.startswith("0x")
                                            else owner_sig))

            nonce = next_nonce(w3, account.address)
            chain_id = self.config.chain_id or w3.eth.chain_id

            try:
                estimated = contract.functions.registerAgentBound(
                    peer_bytes, owner_addr, parent_addr, sig_bytes,
                ).estimate_gas({"from": account.address})
                gas_limit = max(int(estimated * 12 // 10), 500_000)
            except Exception:
                gas_limit = 1_500_000

            tx = contract.functions.registerAgentBound(
                peer_bytes, owner_addr, parent_addr, sig_bytes,
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

    def sign_binding(self, signer_key: str, agent: str, parent: str = "",
                     nonce: int = 0) -> str:
        """EIP-712 ``OwnerBinding{agent, parent, nonce}`` signed by
        ``signer_key``. A registered parent agent signs its child's binding
        with this, so agents spawn registered children without the owner
        wallet; the child lands in the parent's household."""
        from eth_account import Account
        from eth_account.messages import encode_typed_data

        w3 = self._get_web3()
        contract = self._get_contract(w3)
        chain_id = self.config.chain_id or w3.eth.chain_id
        message = encode_typed_data(full_message={
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "OwnerBinding": [
                    {"name": "agent", "type": "address"},
                    {"name": "parent", "type": "address"},
                    {"name": "nonce", "type": "uint256"},
                ],
            },
            "primaryType": "OwnerBinding",
            "domain": {"name": "AutonetSubstrate", "version": "1",
                       "chainId": int(chain_id),
                       "verifyingContract": contract.address},
            "message": {"agent": w3.to_checksum_address(agent),
                        "parent": (w3.to_checksum_address(parent)
                                   if parent else ZERO_ADDRESS),
                        "nonce": int(nonce)},
        })
        signed = Account.from_key(signer_key).sign_message(message)
        return "0x" + bytes(signed.signature).hex()

    async def revoke_agent(self, agent: str, owner_sig: str | bytes,
                           relayer_key: str) -> dict[str, Any]:
        """Relay the owner wallet's ``RevokeAgent{agent}`` signature. Revokes
        the agent and every descendant, permanently. ``relayer_key`` only
        pays gas (any agent key works); the owner's signature authorizes."""
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(relayer_key)
            sig_bytes = (owner_sig if isinstance(owner_sig, bytes)
                         else bytes.fromhex(owner_sig[2:] if owner_sig.startswith("0x")
                                            else owner_sig))
            fn = contract.functions.revokeAgent(w3.to_checksum_address(agent), sig_bytes)
            try:
                gas_limit = max(int(fn.estimate_gas({"from": account.address}) * 12 // 10),
                                200_000)
            except Exception:
                gas_limit = 500_000
            tx = fn.build_transaction({
                "from": account.address,
                "nonce": next_nonce(w3, account.address),
                "gas": gas_limit,
                "gasPrice": w3.eth.gas_price,
                "chainId": self.config.chain_id or w3.eth.chain_id,
            })
            signed = account.sign_transaction(tx)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            tx_hash = w3.eth.send_raw_transaction(raw)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt.status == 1:
                log.info("Agent %s revoked (with its subtree) tx=%s", agent, tx_hash.hex())
                return {"success": True, "tx_hash": tx_hash.hex(), "agent": agent}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hash.hex()}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to revoke agent %s", agent)
            return {"success": False, "error": str(e)}

    async def is_live(self, address: str) -> bool:
        """Registered, not revoked, no revoked ancestor."""
        w3 = self._get_web3()
        contract = self._get_contract(w3)
        return bool(contract.functions.isLive(w3.to_checksum_address(address)).call())

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

            nonce = next_nonce(w3, account.address)
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
            nonce = next_nonce(w3, account.address)
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
            nonce = next_nonce(w3, account.address)
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
    # Household claim (write)
    # ------------------------------------------------------------------

    def _send_signed(self, w3: Any, account: Any, call: Any, *, gas_floor: int) -> dict[str, Any]:
        try:
            estimated = call.estimate_gas({"from": account.address})
            gas_limit = max(int(estimated * 12 // 10), gas_floor)
        except Exception:
            gas_limit = 2_000_000
        tx = call.build_transaction({
            "from": account.address,
            "nonce": next_nonce(w3, account.address),
            "gas": gas_limit,
            "gasPrice": w3.eth.gas_price,
            "chainId": self.config.chain_id or w3.eth.chain_id,
        })
        signed = account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = w3.eth.send_raw_transaction(raw)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status == 1:
            return {"success": True, "tx_hash": tx_hash.hex()}
        return {"success": False, "error": "Transaction reverted", "tx_hash": tx_hash.hex()}

    async def claim(
        self,
        private_key: str,
        earner: str,
        epoch_id_hash: str | bytes,
        tokens: list[str],
        amounts: list[int],
        proof: list[str | bytes],
        seed: int = 0,
    ) -> dict[str, Any]:
        """Sign and submit ``claim(earner, epochIdHash, tokens, amounts,
        seed, proof)`` (``seed``: the leaf's part of the REP floor, value
        units), which pays the whole basket in the same transaction, then
        ``withdraw(earner, token)`` only for a token still owed after it (one
        whose transfer the claim deferred: paused, frozen, or deferred by an
        earlier claim). Normally that is one transaction.

        Decision 2026-10-27: the household's owner wallet is the earner and
        anyone may submit; ``private_key`` only signs and pays gas. The
        leaf under the anchor's claimRoot commits (earner, tokens,
        amounts, seed), tokens strictly ascending, so ``proof`` must be built
        over the same leaf (nodes/common/mint_merkle.py claim_merkle_proof).
        Once per (earner, epoch) on chain (``claimedEpoch``); the epoch
        must be anchored. A token whose withdrawal fails is reported in
        ``withdraw_errors`` and blocks none of the others; it stays owed
        on chain, so a later call (or ``withdraw``) retries it. ``owed``
        lists every token still owed after this call (``{token: amount}``).
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)
            who = w3.to_checksum_address(earner)
            eid = (epoch_id_hash if isinstance(epoch_id_hash, bytes)
                   else _to_bytes32(epoch_id_hash))
            proof_bytes = [(p if isinstance(p, bytes) else _to_bytes32(p)) for p in proof]
            token_list = [w3.to_checksum_address(t) for t in tokens]
            out: dict[str, Any] = {"success": True}
            if not contract.functions.claimedEpoch(who, eid).call():
                out = self._send_signed(w3, account, contract.functions.claim(
                    who, eid, token_list, [int(a) for a in amounts], int(seed),
                    proof_bytes),
                    gas_floor=500_000)
                if not out.get("success"):
                    return out
            # The claim paid every token it could; only what is still owed
            # (deferred by this claim or an earlier one) needs a withdraw.
            out["withdrawn"], out["withdraw_errors"], out["owed"] = {}, {}, {}
            for t in token_list:
                try:
                    if int(contract.functions.owed(who, t).call()) <= 0:
                        continue
                    r = self._send_signed(w3, account, contract.functions.withdraw(who, t),
                                          gas_floor=150_000)
                    if r.get("success"):
                        out["withdrawn"][t.lower()] = r["tx_hash"]
                    else:
                        out["withdraw_errors"][t.lower()] = r.get("error", "reverted")
                except Exception as e:  # noqa: BLE001 (one token never blocks the rest)
                    out["withdraw_errors"][t.lower()] = str(e)
                try:
                    left = int(contract.functions.owed(who, t).call())
                    if left > 0:
                        out["owed"][t.lower()] = left
                except Exception:  # noqa: BLE001 (a read; the claim already landed)
                    pass
            return out
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to claim the household's leaf on chain")
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
        """Read an agent's on-chain record from Substrate.sol's ``agents`` map.

        Returns None when the address is simply not registered. An RPC/ABI
        failure PROPAGATES, so callers can tell "no record" from "could not
        read"; the two used to collapse into the same silent None.
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            addr = w3.to_checksum_address(address)
            registered_at, active = contract.functions.agents(addr).call()[:2]
            if registered_at == 0:
                return None
            peer_id = contract.functions.getAgentPeerId(addr).call()
            return {
                "agent_address": addr,
                "peer_id": peer_id.hex() if isinstance(peer_id, bytes) else str(peer_id),
                "registered_at": registered_at,
                "active": active,
            }
        except Exception as e:
            log.warning("Failed to read agent record for %s: %s", address, e)
            raise

    async def get_agent_peer_id(self, address: str) -> bytes | None:
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            return contract.functions.getAgentPeerId(
                w3.to_checksum_address(address)
            ).call()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # The value index
    # ------------------------------------------------------------------

    def _token(self, w3: Any, token: str) -> Any:
        return w3.eth.contract(address=w3.to_checksum_address(token), abi=ERC20_ABI)

    def _index_candidates(self, w3: Any, contract: Any) -> list[str]:
        """Every token the DAO Registry names under ``jurisdiction.parity.``
        (any value), plus every pool token Substrate has listed. Lowercase
        0x, sorted. The Registry is read through its key enumeration; a
        Registry that cannot enumerate leaves the pool tokens only."""
        out: set[str] = set()
        try:
            reg = w3.eth.contract(
                address=w3.to_checksum_address(contract.functions.registry().call()),
                abi=REGISTRY_ENUM_ABI)
            for key in reg.functions.getAllKeys().call():
                key = str(key)
                if key.startswith(PARITY_KEY_PREFIX):
                    t = key[len(PARITY_KEY_PREFIX):].strip().lower()
                    if len(t) == 42 and t.startswith("0x"):
                        out.add(t)
        except Exception as e:  # noqa: BLE001 (no enumeration: pool tokens only)
            log.debug("Registry key enumeration failed: %s", e)
        try:
            out.update(str(t).lower() for t in contract.functions.poolTokens().call())
        except Exception as e:  # noqa: BLE001
            log.debug("poolTokens read failed: %s", e)
        return sorted(out)

    def token_info(self, w3: Any, contract: Any, token: str) -> dict[str, Any]:
        """``{token, symbol, decimals, parity, accepted}`` for one token, as
        Substrate and the token itself say now. ``decimals`` is the token's
        own ``decimals()``; ``parity`` is the index's (0 = not in it)."""
        t = self._token(w3, token)
        try:
            symbol = str(t.functions.symbol().call())
        except Exception:  # noqa: BLE001 (symbol is optional in ERC-20)
            symbol = ""
        try:
            decimals: int | None = int(t.functions.decimals().call())
        except Exception:  # noqa: BLE001
            decimals = None
        cs = w3.to_checksum_address(token)
        return {
            "token": str(token).lower(),
            "symbol": symbol,
            "decimals": decimals,
            "parity": str(int(contract.functions.parityOf(cs).call())),
            "accepted": bool(contract.functions.acceptsToken(cs).call()),
        }

    async def index_tokens(self) -> list[dict[str, Any]]:
        """The tokens services can be priced and paid in now: every token
        the value index lists that Substrate accepts, each
        ``{token, symbol, decimals, parity}`` (``parity`` a decimal string:
        the value of ``amount`` base units is ``amount * parity / 1e30``
        value units, 1e6 value units being one unit of the index). Raises
        when the chain is unreadable."""
        w3 = self._get_web3()
        contract = self._get_contract(w3)
        out = []
        for t in self._index_candidates(w3, contract):
            info = self.token_info(w3, contract, t)
            if info.pop("accepted"):
                out.append(info)
        return out

    async def accepts_token(self, token: str) -> bool:
        """``Substrate.acceptsToken(token)``: a payment in it can be made
        now. False on a read failure or a malformed address."""
        try:
            w3 = self._get_web3()
            return bool(self._get_contract(w3).functions.acceptsToken(
                w3.to_checksum_address(token)).call())
        except Exception as e:  # noqa: BLE001
            log.debug("acceptsToken(%s) failed: %s", token, e)
            return False

    async def value_of(self, token: str, amount: int) -> int:
        """``Substrate.valueOf(token, amount)``: the value of ``amount`` at
        today's index, in 6-decimal value units (0 outside the index)."""
        w3 = self._get_web3()
        return int(self._get_contract(w3).functions.valueOf(
            w3.to_checksum_address(token), int(amount)).call())

    def _balances(self, w3: Any, contract: Any, address: str,
                  tokens: list[str]) -> list[dict[str, Any]]:
        out = []
        cs = w3.to_checksum_address(address)
        for t in tokens:
            info = self.token_info(w3, contract, t)
            try:
                bal = int(self._token(w3, t).functions.balanceOf(cs).call())
            except Exception as e:  # noqa: BLE001
                log.debug("balanceOf(%s) on %s failed: %s", address, t, e)
                continue
            out.append({**info, "balance": str(bal)})
        return out

    async def get_agent_balances(self, address: str) -> dict[str, Any] | None:
        """An address's money: its balance of every index and pool token
        (``tokens``: ``{token, symbol, decimals, parity, accepted,
        balance}``, base units as decimal strings), its native gas balance,
        and the value it has claimed from the pools (``earned_value``,
        6-decimal value units: what the DAO's REP claim reads)."""
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            addr = w3.to_checksum_address(address)
            tokens = self._balances(w3, contract, addr,
                                    self._index_candidates(w3, contract))
            return {
                "address": addr,
                "tokens": tokens,
                "earned_value": str(int(contract.functions.earnedValue(addr).call())),
                "gas_balance": str(w3.eth.get_balance(addr)),
            }
        except Exception as e:
            log.debug("Failed to read balances for %s: %s", address, e)
            return None

    async def get_claim_status(self, earner: str) -> dict[str, Any] | None:
        """What a household's earner (its owner wallet) has from the pools:
        ``earned_value`` (claimed, as value), ``seeded_value`` (the part of
        it the REP floor seeded; "0" when unread) and ``owed`` per pool token
        (claimed and not yet paid: a claim pays its basket at once, so this
        is what a failed transfer deferred, a paused or frozen token, until
        it is withdrawn: ``{token, symbol, decimals, amount}``, only tokens
        with something owed)."""
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            who = w3.to_checksum_address(earner)
            owed = []
            for t in sorted(str(x).lower() for x in contract.functions.poolTokens().call()):
                amount = int(contract.functions.owed(who, w3.to_checksum_address(t)).call())
                if amount > 0:
                    info = self.token_info(w3, contract, t)
                    owed.append({"token": t, "symbol": info["symbol"],
                                 "decimals": info["decimals"], "amount": str(amount)})
            out: dict[str, Any] = {
                "earner": who,
                "earned_value": str(int(contract.functions.earnedValue(who).call())),
                "seeded_value": "0",
                "owed": owed}
            try:
                out["seeded_value"] = str(int(contract.functions.seededValue(who).call()))
            except Exception as e:  # noqa: BLE001 (a read; the rest stands)
                log.debug("seededValue for %s unread: %s", earner, e)
            # The basket so far: every token's total claimed by the earner
            # (ClaimedToken logs), a read failure leaving it out.
            try:
                totals = self._claimed_totals(w3, contract, who)
                rows = []
                for t in sorted(totals):
                    info = self.token_info(w3, contract, t)
                    rows.append({"token": t, "symbol": info["symbol"],
                                 "decimals": info["decimals"],
                                 "amount": str(totals[t])})
                out["claimed"] = rows
            except Exception as e:  # noqa: BLE001
                log.debug("claimed totals for %s unread: %s", earner, e)
            return out
        except Exception as e:  # noqa: BLE001
            log.debug("Failed to read claim status for %s: %s", earner, e)
            return None

    def _claimed_totals(self, w3: Any, contract: Any, earner: str) -> dict[str, int]:
        """Per token, what ``earner`` has claimed from the pools: the sum of
        its ``ClaimedToken`` logs. Followed incrementally per (chain,
        Substrate, earner): the first read scans from the jurisdiction's
        deploy block, later ones only the blocks since. A failed read keeps
        what was summed so far for the next call."""
        from web3 import Web3
        key = (str(getattr(self.config, "rpc_url", "") or ""),
               str(contract.address).lower(), earner.lower())
        with _CLAIMED_LOCK:
            state = _CLAIMED_CACHE.get(key)
            if state is None:
                start = int(getattr(self.config, "deploy_block", 0) or 0)
                state = {"next": start, "totals": {}}
                _CLAIMED_CACHE[key] = state
            topic0 = "0x" + Web3.keccak(
                text="ClaimedToken(address,bytes32,address,uint256)").hex().removeprefix("0x")
            topic_earner = "0x" + "00" * 12 + earner.lower().removeprefix("0x")
            event = contract.events.ClaimedToken()
            last = int(w3.eth.block_number)
            if int(state["next"]) > last + 1:
                # A chain behind what was read (a local chain restarted):
                # sum again from the start.
                state["next"] = int(getattr(self.config, "deploy_block", 0) or 0)
                state["totals"] = {}
            block = int(state["next"])
            while block <= last:
                end = min(last, block + CHANNEL_LOG_CHUNK - 1)
                logs = w3.eth.get_logs({"address": contract.address, "fromBlock": block,
                                        "toBlock": end, "topics": [topic0, topic_earner]})
                for lg in logs:
                    args = event.process_log(lg)["args"]
                    t = str(args["token"]).lower()
                    state["totals"][t] = state["totals"].get(t, 0) + int(args["amount"])
                block = end + 1
                state["next"] = block
            return dict(state["totals"])

    async def withdraw_owed(self, private_key: str, earner: str,
                            token: str) -> dict[str, Any]:
        """``Substrate.withdraw(earner, token)`` signed by ``private_key``
        (anyone may send it; it pays the earner what it is owed in that
        token). Refused when nothing is owed, and dry-run first so a token
        still paused or frozen costs no gas. ``{success, tx_hash, amount}``
        or ``{success: False, error}``."""
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)
            who = w3.to_checksum_address(earner)
            t = w3.to_checksum_address(token)
            owed = int(contract.functions.owed(who, t).call())
            if owed <= 0:
                return {"success": False, "error": "nothing is owed in this token",
                        "amount": "0"}
            call = contract.functions.withdraw(who, t)
            try:
                call.call({"from": account.address})
            except Exception as e:  # noqa: BLE001
                return {"success": False, "amount": str(owed),
                        "error": f"the withdrawal would fail now (the token may be "
                                 f"paused or the earner frozen): {e}"}
            out = self._send_signed(w3, account, call, gas_floor=150_000)
            out["amount"] = str(owed)
            out["token"] = str(token).lower()
            return out
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:  # noqa: BLE001
            log.warning("withdraw(%s, %s) failed: %s", earner, token, e)
            return {"success": False, "error": str(e)}

    def _service_payments_to(self, w3: Any, contract: Any, recipient: str,
                             from_block: int) -> dict[str, int]:
        """Gross paid to ``recipient`` directly (``ServicePayment`` logs),
        per token, read in chunks of CHANNEL_LOG_CHUNK blocks."""
        from web3 import Web3
        topic0 = "0x" + Web3.keccak(
            text="ServicePayment(address,address,address,uint256,bytes32)"
        ).hex().removeprefix("0x")
        topic_to = "0x" + "00" * 12 + recipient.lower().removeprefix("0x")
        event = contract.events.ServicePayment()
        out: dict[str, int] = {}
        last = int(w3.eth.block_number)
        block = max(0, int(from_block))
        while block <= last:
            end = min(last, block + CHANNEL_LOG_CHUNK - 1)
            for lg in w3.eth.get_logs({"address": contract.address, "fromBlock": block,
                                       "toBlock": end, "topics": [topic0, None, topic_to]}):
                args = event.process_log(lg)["args"]
                t = str(args["token"]).lower()
                out[t] = out.get(t, 0) + int(args["amount"])
            block = end + 1
        return out

    async def get_provider_record(self, address: str) -> dict[str, Any] | None:
        """Read a service provider's VERIFIABLE track record.

        Services carry no reputation, no mint and no substrate standing:
        remote execution is unknowable in principle, so there is no honest
        quality score to show (docs/services_market.md). What IS honest is
        what the chain already proves about a counterparty:

          ``service_earnings``  per token, the gross this address has been
                                paid for services: its direct payments
                                (``ServicePayment``) and its payouts from
                                buyers' credits (``ChannelPaid``), from the
                                jurisdiction's deploy block (``deploy_block``)
                                on. A list of ``{token, symbol, decimals,
                                amount}``.
          ``earned_value``      what the address has claimed from the tool
                                pools, as value (the same address authoring
                                tools the network uses).
          ``registered_at``     identity tenure. An address that has been
                                serving for a year is a different
                                counterparty from one registered this morning.

        They are consequences of payments, which makes them costly to
        inflate rather than impossible: a provider can pay itself and lose
        the service fee on every payment. This is a track record, not a
        rating.
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            addr = w3.to_checksum_address(address)
            start = int(getattr(self.config, "deploy_block", 0) or 0)
            paid = self._service_payments_to(w3, contract, addr, start)
            try:
                smc = ServiceMarketClient(self.config)
                if smc.channel_available:
                    for p in await smc.channel_payments(start, provider=addr) or []:
                        t = str(p.get("token") or "").lower()
                        if t:
                            paid[t] = paid.get(t, 0) + int(p["amount"])
            except Exception as e:  # noqa: BLE001 (direct payments still count)
                log.debug("provider_record: channel payouts unread: %s", e)
            earnings = []
            for t in sorted(paid):
                info = self.token_info(w3, contract, t)
                earnings.append({"token": t, "symbol": info["symbol"],
                                 "decimals": info["decimals"], "amount": str(paid[t])})
            try:
                earned_value = int(contract.functions.earnedValue(addr).call())
            except Exception:                              # noqa: BLE001
                earned_value = 0
            try:
                registered_at = int(contract.functions.agents(addr).call()[0])
            except Exception:                              # noqa: BLE001
                registered_at = 0
            return {
                "address": addr,
                "service_earnings": earnings,
                "earned_value": str(earned_value),
                "registered_at": registered_at,
            }
        except Exception as e:                             # noqa: BLE001
            log.debug("Failed to read provider record for %s: %s", address, e)
            return None

    async def get_fleet_voice(self, owner: str) -> dict[str, Any] | None:
        """An owner's household on chain: the owner wallet's and every bound
        agent's balance of each index token, the value each has claimed
        from the pools, and the household's REP over REP supply (the
        weight the close reads, from RepToken).

        Money is per token in base units (``tokens`` rows as
        ``get_agent_balances`` gives them; ``fleet_tokens`` the household's
        total per token). Value is in 6-decimal value units
        (``earned_value``, ``fleet_earned_value``, ``total_earned_value``
        network wide). REP balances and supply are raw (18 decimals); the
        close pins them to the previous anchor, this reads the latest.
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            owner_addr = w3.to_checksum_address(owner)
            tokens = self._index_candidates(w3, contract)
            owner_tokens = self._balances(w3, contract, owner_addr, tokens)
            fleet: dict[str, int] = {r["token"]: int(r["balance"]) for r in owner_tokens}
            owner_value = int(contract.functions.earnedValue(owner_addr).call())
            fleet_value = owner_value
            agents: list[dict[str, Any]] = []
            count = contract.functions.registeredAgentCount().call()
            for i in range(count):
                try:
                    addr = contract.functions.getRegisteredAgent(i).call()
                    bound = contract.functions.agentOwner(addr).call()
                    if str(bound).lower() != str(owner_addr).lower():
                        continue
                    rows = self._balances(w3, contract, addr, tokens)
                    value = int(contract.functions.earnedValue(addr).call())
                    agents.append({"agent_id": str(addr), "tokens": rows,
                                   "earned_value": str(value)})
                    for r in rows:
                        fleet[r["token"]] = fleet.get(r["token"], 0) + int(r["balance"])
                    fleet_value += value
                except Exception as e:
                    log.debug("fleet_voice: agent %d read failed: %s", i, e)
                    continue
            rep_balance_raw = rep_total_raw = None
            rep_addr = getattr(self.config, "rep_token_address", "") or ""
            if rep_addr:
                try:
                    rep = w3.eth.contract(
                        address=w3.to_checksum_address(rep_addr), abi=[
                            {"name": "balanceOf", "type": "function",
                             "stateMutability": "view",
                             "inputs": [{"name": "a", "type": "address"}],
                             "outputs": [{"name": "", "type": "uint256"}]},
                            {"name": "totalSupply", "type": "function",
                             "stateMutability": "view", "inputs": [],
                             "outputs": [{"name": "", "type": "uint256"}]}])
                    rep_balance_raw = int(rep.functions.balanceOf(owner_addr).call())
                    for a in agents:
                        rep_balance_raw += int(rep.functions.balanceOf(
                            w3.to_checksum_address(a["agent_id"])).call())
                    rep_total_raw = int(rep.functions.totalSupply().call())
                except Exception as e:
                    log.debug("fleet_voice: REP read failed: %s", e)
                    rep_balance_raw = rep_total_raw = None
            info = {r["token"]: r for r in owner_tokens}
            return {
                "owner": str(owner_addr),
                "household_rep_raw": rep_balance_raw,
                "rep_total_supply_raw": rep_total_raw,
                "owner_tokens": owner_tokens,
                "owner_earned_value": str(owner_value),
                "fleet_tokens": [
                    {**{k: v for k, v in info.get(t, {"token": t}).items()
                        if k != "balance"}, "balance": str(b)}
                    for t, b in sorted(fleet.items())],
                "fleet_earned_value": str(fleet_value),
                "total_earned_value": str(int(
                    contract.functions.totalEarnedValue().call())),
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

        Returns the anchor count, the latest anchor hash and epoch root,
        the registered agent count, the value claimed from the pools so far
        (``total_earned_value``, 6-decimal value units) and each pool
        (``pools``: per token, the tool share collected, what has been
        claimed and the cap claims are held to, base units as decimal
        strings).
        """
        try:
            w3 = self._get_web3()
            contract = self._get_contract(w3)
            anchor_count = contract.functions.anchorCount().call()
            latest_anchor = contract.functions.latestAnchorHash().call()
            latest_root = contract.functions.latestEpochRoot().call()
            agent_count = contract.functions.registeredAgentCount().call()
            pools = []
            for t in sorted(str(x).lower() for x in contract.functions.poolTokens().call()):
                cs = w3.to_checksum_address(t)
                info = self.token_info(w3, contract, t)
                pools.append({
                    **info,
                    "tool_share_total": str(int(contract.functions.toolShareTotal(cs).call())),
                    "claimed_total": str(int(contract.functions.claimedTotal(cs).call())),
                    "pool_cap": str(int(contract.functions.poolCap(cs).call())),
                })
            return {
                "anchor_count": anchor_count,
                "latest_anchor_hash": latest_anchor.hex() if isinstance(latest_anchor, bytes) else str(latest_anchor),
                "latest_epoch_root": latest_root.hex() if isinstance(latest_root, bytes) else str(latest_root),
                "total_earned_value": str(int(contract.functions.totalEarnedValue().call())),
                "pools": pools,
                "registered_agent_count": agent_count,
            }
        except Exception as e:
            log.debug("Failed to read substrate state: %s", e)
            return None

    # ------------------------------------------------------------------
    # Service payments (Substrate.payForService)
    # ------------------------------------------------------------------

    def _send_tx(self, w3: Any, account: Any, call: Any, *, mult: int,
                 floor: int) -> tuple[Any, str]:
        """Estimate (``mult``/10 of it, at least ``floor``), sign, send and
        wait for the receipt. A call whose estimate fails is never sent:
        raises ``WouldRevert`` with the reason."""
        try:
            estimated = call.estimate_gas({"from": account.address})
        except Exception as exc:  # noqa: BLE001 (a revert or a failed read)
            raise WouldRevert(revert_reason(exc)) from exc
        tx = call.build_transaction({
            "from": account.address,
            "nonce": next_nonce(w3, account.address),
            "gas": max(int(estimated * mult // 10), floor),
            "gasPrice": w3.eth.gas_price,
            "chainId": self.config.chain_id or w3.eth.chain_id,
        })
        signed = account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = w3.eth.send_raw_transaction(raw)
        return w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120), tx_hash.hex()

    async def pay_for_service(
        self,
        private_key: str,
        token: str,
        recipient: str,
        amount: int,
        request_id: str | bytes,
    ) -> dict[str, Any]:
        """Pay ``recipient`` ``amount`` (gross, in ``token``'s base units)
        for a service: ``Substrate.payForService(token, recipient, amount,
        requestId)`` from the key's address, the payer.

        ``token`` must be in the value index (Substrate refuses it
        otherwise; this checks ``acceptsToken`` first and sends nothing when
        it is not). Substrate pulls the gross from the payer, so the payer
        approves Substrate on the token first: an ``approve`` is sent when
        the allowance falls short (``approve_tx_hash``). The recipient
        receives the gross less the service fee. ``request_id`` (bytes32)
        rides the ServicePayment event so the payment is matchable off
        chain.

        Gas: the estimate x1.3, at least PAY_GAS_FLOOR (APPROVE_GAS_FLOOR
        for the approve). A transaction whose estimate fails is not sent.
        Returns ``{success, tx_hash, request_id, approve_tx_hash}`` or
        ``{success: False, error}``.
        """
        try:
            from eth_account import Account

            w3 = self._get_web3()
            contract = self._get_contract(w3)
            account = Account.from_key(private_key)
            token_addr = w3.to_checksum_address(token)
            recipient_addr = w3.to_checksum_address(recipient)
            req = (request_id if isinstance(request_id, bytes)
                   else _to_bytes32(request_id))
            if not contract.functions.acceptsToken(token_addr).call():
                return {"success": False,
                        "error": f"token {token_addr} is not in the value index "
                                 "(Substrate does not accept it)"}
            erc20 = self._token(w3, token_addr)
            approve_tx_hash = None
            if int(erc20.functions.allowance(account.address,
                                             contract.address).call()) < int(amount):
                receipt, h = self._send_tx(
                    w3, account, erc20.functions.approve(contract.address, int(amount)),
                    mult=13, floor=APPROVE_GAS_FLOOR)
                if receipt.status != 1:
                    return {"success": False, "error": "approve reverted", "tx_hash": h}
                approve_tx_hash = h
            receipt, tx_hex = self._send_tx(
                w3, account, contract.functions.payForService(
                    token_addr, recipient_addr, int(amount), req),
                mult=13, floor=PAY_GAS_FLOOR)
            if receipt.status == 1:
                return {
                    "success": True,
                    "tx_hash": tx_hex,
                    "request_id": req.hex() if isinstance(req, bytes) else str(req),
                    "approve_tx_hash": approve_tx_hash,
                }
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hex, "approve_tx_hash": approve_tx_hash}
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
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
        token: str,
        payer: str | None = None,
        tx_hash: str | bytes | None = None,
    ) -> dict[str, Any]:
        """Verify a claimed on-chain service payment, cheaply.

        Etherlink caps log-range scans, so this NEVER scans a block range:
        a ``tx_hash`` is REQUIRED. It fetches that one receipt and decodes
        the ``ServicePayment`` event(s) from its logs (Substrate is the
        emitter), then checks the claimed facts:

          - the event was emitted by this jurisdiction's Substrate (a log
            of the same shape from any other contract is ignored);
          - ``requestId`` matches ``request_id``;
          - ``recipient`` matches ``recipient``;
          - ``token`` matches ``token`` (the price list's token: an amount
            is only comparable in the token it is priced in, so a payment
            in another token is refused whatever its number);
          - ``amount`` (gross, in that token's base units) is >=
            ``min_amount``;
          - if ``payer`` is given, the event's payer matches it.

        Returns a dict:
          ``{verified: bool, reason: str, payer, recipient, token, amount,
             request_id, tx_hash}``. ``verified`` is False (with a reason)
          on any mismatch, a missing/absent event, a reverted tx, or a read
          failure. ``amount`` is the gross as recorded in the event.
        """
        result: dict[str, Any] = {
            "verified": False,
            "reason": "",
            "payer": None,
            "recipient": None,
            "token": None,
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

            # Only Substrate's own events: any contract can emit a log with
            # ServicePayment's signature.
            events = [ev for ev in events
                      if str(ev.get("address") or "").lower() == contract.address.lower()]
            if not events:
                result["reason"] = "no ServicePayment event in receipt"
                return result
            want_token = w3.to_checksum_address(token)

            # Find the event matching the claimed requestId + recipient.
            for ev in events:
                args = ev["args"]
                ev_req = args["requestId"]
                ev_req_b = (ev_req if isinstance(ev_req, bytes)
                            else _to_bytes32(str(ev_req)))
                ev_recipient = w3.to_checksum_address(args["recipient"])
                ev_payer = w3.to_checksum_address(args["payer"])
                ev_amount = int(args["amount"])
                ev_token = w3.to_checksum_address(args["token"])

                if ev_req_b != want_req:
                    continue
                if ev_recipient != recipient_addr:
                    continue

                result.update({
                    "payer": ev_payer,
                    "recipient": ev_recipient,
                    "token": ev_token.lower(),
                    "amount": ev_amount,
                    "request_id": ev_req_b.hex(),
                })
                if ev_token != want_token:
                    result["reason"] = (
                        f"paid in token {ev_token}, but the price is in {want_token}")
                    return result
                if ev_amount < int(min_amount):
                    result["reason"] = (
                        f"amount {ev_amount} < min_amount {int(min_amount)}")
                    return result
                if payer is not None:
                    if ev_payer != w3.to_checksum_address(payer):
                        result["reason"] = "payer mismatch"
                        return result
                # When it was paid: the gate refuses a payment older than its
                # replay horizon, so a used-up request id need not be kept
                # forever (ServiceStore.REPLAY_HORIZON_S).
                try:
                    block_no = (receipt.get("blockNumber") if isinstance(receipt, dict)
                                else getattr(receipt, "blockNumber", None))
                    if block_no is not None:
                        result["paid_at"] = int(w3.eth.get_block(block_no)["timestamp"])
                except Exception:  # noqa: BLE001 - the gate then has no age to check
                    pass
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

# The custom errors ServiceMarket.sol reverts with (its ABI here carries none,
# so web3 reports only the 4-byte selector). Named for the revert reason.
_SERVICE_MARKET_ERRORS = (
    "NotRegisteredAgent()", "SpecDigestRequired()", "NoSuchService()",
    "NotServiceProvider()", "ServiceNotActive()", "ProviderRequired()",
    "ZeroDeposit()", "ChannelNotOpen()", "ChannelNotClosing()",
    "NotProvider()", "NotClient()", "BadVoucherSignature()",
    "WindowNotElapsed()", "NotGovernor()", "ZeroWindow()", "NothingToSettle()",
    "TokenNotAccepted(address)", "ShortReceipt(uint256,uint256)",
    # Substrate's, for a direct payment.
    "NotInIndex(address)", "ZeroAddress()", "ZeroAmount()",
    "ShortReceipt(address,uint256,uint256)", "BadTokenDecimals(address)",
)
_error_names: dict[str, str] = {}


class WouldRevert(Exception):
    """Gas estimation says the transaction fails: it was not sent."""


def revert_reason(exc: BaseException) -> str:
    """A readable reason for a failed estimate or call: the custom error's
    name when its selector is a known ServiceMarket error, else the node's
    message."""
    if not _error_names:
        try:
            from eth_utils import keccak
            for sig in _SERVICE_MARKET_ERRORS:
                _error_names["0x" + keccak(text=sig)[:4].hex()] = sig.split("(")[0]
        except Exception:  # noqa: BLE001 (no eth_utils: raw messages only)
            pass
    candidates = [getattr(exc, "data", None), *getattr(exc, "args", ())]
    for item in candidates:
        text = str(item or "")
        for selector, name in _error_names.items():
            if selector in text.lower():
                return name
    message = getattr(exc, "message", None) or str(exc)
    return str(message) or exc.__class__.__name__


@_off_the_loop
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
      - Substrate (the value index, the fee): ``config.substrate_address``
        (falls back to ``rpb_contract_address``).

    A channel's token is any token of the value index, fixed when it opens;
    the deposit is approved on that token for the PaymentChannel.
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

    def _token_contract(self, w3, token: str):
        from web3 import Web3
        return w3.eth.contract(address=Web3.to_checksum_address(token), abi=ERC20_ABI)

    def _substrate_contract(self, w3=None):
        from web3 import Web3
        if w3 is None:
            w3 = self._get_web3()
        addr = self._substrate_address()
        if not addr:
            raise ValueError("substrate_address not configured")
        return w3.eth.contract(address=Web3.to_checksum_address(addr), abi=SUBSTRATE_ABI)

    async def accepts_token(self, token: str) -> bool:
        """``Substrate.acceptsToken(token)`` (False on a read failure)."""
        try:
            w3 = self._get_web3()
            return bool(self._substrate_contract(w3).functions.acceptsToken(
                w3.to_checksum_address(token)).call())
        except Exception as e:  # noqa: BLE001
            log.debug("acceptsToken(%s) failed: %s", token, e)
            return False

    async def value_of(self, token: str, amount: int) -> int | None:
        """``Substrate.valueOf(token, amount)``: the value of ``amount`` of
        ``token`` at today's index in 6-decimal value units (0 when the
        token is not in it); None on a read failure."""
        try:
            w3 = self._get_web3()
            return int(self._substrate_contract(w3).functions.valueOf(
                w3.to_checksum_address(token), int(amount)).call())
        except Exception as e:  # noqa: BLE001
            log.debug("valueOf(%s) failed: %s", token, e)
            return None

    @staticmethod
    def _send_signed(w3, account, func, chain_id: int, gas_floor: int):
        """Estimate gas (×1.2, floored), build/sign/send, wait for receipt.

        Shared write path — mirrors OnChainService's per-method inline
        pattern. Returns (receipt, tx_hash_hex). A transaction whose
        estimate fails is never sent (it would only burn gas on a revert):
        raises ``WouldRevert`` with the reason. The nonce is taken after
        the estimate, so a write that is not sent leaves no gap.
        """
        try:
            estimated = func.estimate_gas({"from": account.address})
        except Exception as exc:  # noqa: BLE001 (a revert or a failed read)
            raise WouldRevert(revert_reason(exc)) from exc
        gas_limit = max(int(estimated * 12 // 10), gas_floor)
        nonce = next_nonce(w3, account.address)
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
        ``ask_amount`` is an optional price that nothing enforces (prices
        are the provider's signed list, which names their token).

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
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
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
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
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
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
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
        token: str,
        amount: int,
    ) -> dict[str, Any]:
        """Approve the PaymentChannel to pull ``amount`` of ``token``
        (``token.approve(channel, amount)``).

        openChannel and topUp pull the deposit from the client with the
        token's ``transferFrom``, so the client approves the channel
        contract on the token first. Signed by the client key.
        """
        try:
            from eth_account import Account
            from web3 import Web3

            w3 = self._get_web3()
            erc20 = self._token_contract(w3, token)
            account = Account.from_key(private_key)
            chain_id = self.config.chain_id or w3.eth.chain_id
            channel_addr = Web3.to_checksum_address(self._payment_channel_address())
            func = erc20.functions.approve(channel_addr, int(amount))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 120_000)
            if receipt.status == 1:
                return {"success": True, "tx_hash": tx_hex}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hex}
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to approve channel deposit on chain")
            return {"success": False, "error": str(e)}

    async def open_channel(
        self,
        private_key: str,
        provider: str,
        token: str,
        deposit: int,
        auto_approve: bool = True,
    ) -> dict[str, Any]:
        """``PaymentChannel.openChannel(provider, token, deposit)``.

        The client escrows ``deposit`` (base units of ``token``, a token of
        the value index; the contract refuses any other) with ``provider``.
        openChannel pulls the deposit with the token's ``transferFrom``, so
        an allowance must exist first: when ``auto_approve`` (default), this
        checks the current allowance and approves the deposit on the token
        when it falls short. Signed by the client key.

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

            token_addr = w3.to_checksum_address(token)
            approve_tx_hash = None
            if auto_approve:
                erc20 = self._token_contract(w3, token_addr)
                channel_addr = Web3.to_checksum_address(
                    self._payment_channel_address())
                try:
                    current = int(erc20.functions.allowance(
                        account.address, channel_addr).call())
                except Exception:
                    current = 0
                if current < int(deposit):
                    ap = await self.approve_channel_deposit(
                        private_key, token_addr, int(deposit))
                    if not ap.get("success"):
                        return {"success": False,
                                "error": f"approve failed: {ap.get('error')}"}
                    approve_tx_hash = ap.get("tx_hash")

            func = channel.functions.openChannel(provider_addr, token_addr, int(deposit))
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
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to open channel on chain")
            return {"success": False, "error": str(e)}

    async def top_up_channel(
        self,
        client_key: str,
        channel_id: int,
        amount: int,
        auto_approve: bool = True,
    ) -> dict[str, Any]:
        """``PaymentChannel.topUp(channelId, amount)``, by the CLIENT, while
        the channel is Open: adds ``amount`` (base units of the channel's
        token) to its deposit (the buyer's credit with that provider).
        topUp pulls with the token's ``transferFrom``, so, as in
        :meth:`open_channel`, an ``approve`` of the channel's token is sent
        first when the allowance falls short (``auto_approve``).

        Returns ``{success, tx_hash, deposit, approve_tx_hash}``; ``deposit``
        is the new deposit, decoded from the ChannelToppedUp event (None if
        it couldn't be decoded)."""
        try:
            from eth_account import Account
            from web3 import Web3

            w3 = self._get_web3()
            channel = self._channel_contract(w3)
            account = Account.from_key(client_key)
            chain_id = self.config.chain_id or w3.eth.chain_id

            approve_tx_hash = None
            if auto_approve:
                ch = channel.functions.getChannel(int(channel_id)).call()
                erc20 = self._token_contract(w3, ch[2])
                channel_addr = Web3.to_checksum_address(
                    self._payment_channel_address())
                try:
                    current = int(erc20.functions.allowance(
                        account.address, channel_addr).call())
                except Exception:
                    current = 0
                if current < int(amount):
                    ap = await self.approve_channel_deposit(client_key, ch[2], int(amount))
                    if not ap.get("success"):
                        return {"success": False,
                                "error": f"approve failed: {ap.get('error')}"}
                    approve_tx_hash = ap.get("tx_hash")

            func = channel.functions.topUp(int(channel_id), int(amount))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 150_000)
            if receipt.status != 1:
                return {"success": False, "error": "Transaction reverted",
                        "tx_hash": tx_hex, "approve_tx_hash": approve_tx_hash}
            deposit = None
            try:
                from web3.logs import DISCARD
                evs = channel.events.ChannelToppedUp().process_receipt(
                    receipt, errors=DISCARD)
                if evs:
                    deposit = int(evs[0]["args"]["deposit"])
            except Exception:
                log.debug("top_up_channel: event decode failed", exc_info=True)
            return {"success": True, "tx_hash": tx_hex, "deposit": deposit,
                    "approve_tx_hash": approve_tx_hash}
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to top up channel on chain")
            return {"success": False, "error": str(e)}

    def sign_voucher(
        self,
        private_key: str,
        channel_id: int,
        token: str,
        cumulative_amount: int,
    ) -> str:
        """Sign a PaymentChannel voucher over (channelId, token,
        cumulativeAmount).

        PURE / offline: builds the EIP-712 digest with
        :func:`build_voucher_digest` (matching the contract's
        ``EIP712("AutonetPaymentChannel","1")`` domain and ``Voucher(uint256
        channelId,address token,uint256 cumulativeAmount)`` typehash) using
        the configured channel address and chain id, then signs the raw
        32-byte digest with ``private_key`` (the CLIENT key). ``token`` is
        the channel's token; ``cumulativeAmount`` the GROSS, MONOTONE
        running total the client authorizes to date, in its base units.

        Returns the 0x-prefixed hex signature ``settle`` and
        ``closeChannel`` accept.
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
            int(channel_id), token, int(cumulative_amount), channel_addr, chain_id)
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

        Reads the channel, rebuilds the EIP-712 digest offline
        (:func:`build_voucher_digest`) over the channel's own token,
        recovers the signer from ``signature``, and verifies: the channel
        exists and is Open (status == 1), the recovered signer equals the
        channel's ``client``, and (advisory) reports whether
        ``cumulative_amount`` exceeds the deposit (the contract caps the
        payout at the deposit, so an over-deposit voucher is not a signature
        failure). The caller compares ``token`` with the token it prices in.

        Returns ``{valid, reason, signer, client, provider, token, deposit,
        status, exceeds_deposit}``. ``valid`` is True only when the signer
        matches the channel client AND the channel is Open.
        """
        result: dict[str, Any] = {
            "valid": False, "reason": "", "signer": None, "client": None,
            "provider": None, "token": None, "deposit": None, "status": None,
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
            channel = self._channel_contract(w3)
            ch = channel.functions.getChannel(int(channel_id)).call()
            client, provider, token, deposit, claimed, closes_at, status = ch[:7]
            digest = build_voucher_digest(
                int(channel_id), token, int(cumulative_amount), channel_addr, chain_id)
            sig = (signature if isinstance(signature, bytes)
                   else bytes.fromhex(signature[2:] if signature.startswith("0x")
                                      else signature))
            # Recover the signer of the raw typed-data digest.
            signer = (Account._recover_hash(digest, signature=sig)
                      if hasattr(Account, "_recover_hash")
                      else Account.recoverHash(digest, signature=sig))
            signer = Web3.to_checksum_address(signer)
            result["signer"] = signer
            result.update({
                "client": Web3.to_checksum_address(client),
                "provider": Web3.to_checksum_address(provider),
                "token": str(token).lower(),
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
        up_to: int | None = None,
    ) -> dict[str, Any]:
        """``PaymentChannel.closeChannel(channelId, cumulativeAmount,
        signature, upTo)``.

        Submitted by the PROVIDER (msg.sender must be the channel provider).
        Ends the channel: pays the provider what of min(cumulative, deposit,
        up_to) ``settle`` has not paid yet, NET of the service fee. ``up_to``
        is what the provider has served (vouchers pay ahead); None claims the
        whole voucher. The client's
        remainder is withdrawable at once (nothing more can be collected,
        so there is no window). ``signature`` is the client's voucher
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
                int(channel_id), int(cumulative_amount), sig,
                int(cumulative_amount if up_to is None else up_to))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 200_000)
            if receipt.status == 1:
                return {"success": True, "tx_hash": tx_hex}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hex}
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to close channel on chain")
            return {"success": False, "error": str(e)}

    async def settle_channel(
        self,
        provider_key: str,
        channel_id: int,
        cumulative_amount: int,
        signature: str | bytes,
        up_to: int | None = None,
    ) -> dict[str, Any]:
        """``PaymentChannel.settle(channelId, cumulativeAmount, signature,
        upTo)``.

        Submitted by the PROVIDER while the channel is Open (or inside the
        window of a close the client started): cashes in the voucher without
        ending the channel, paying min(cumulative, deposit, up_to) minus what
        was already paid, NET of the service fee. ``up_to`` is what the
        provider has served (vouchers pay ahead); None claims the whole
        voucher. A call that adds nothing reverts (NothingToSettle).

        Returns ``{success, tx_hash, paid, paid_total}``: the gross amount
        paid now and the channel's cumulative paid, decoded from the
        ChannelPaid event (None if it couldn't be decoded)."""
        try:
            from eth_account import Account

            w3 = self._get_web3()
            channel = self._channel_contract(w3)
            account = Account.from_key(provider_key)
            chain_id = self.config.chain_id or w3.eth.chain_id
            sig = (signature if isinstance(signature, bytes)
                   else bytes.fromhex(signature[2:] if signature.startswith("0x")
                                      else signature))
            func = channel.functions.settle(
                int(channel_id), int(cumulative_amount), sig,
                int(cumulative_amount if up_to is None else up_to))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 200_000)
            if receipt.status != 1:
                return {"success": False, "error": "Transaction reverted",
                        "tx_hash": tx_hex}
            paid = paid_total = None
            try:
                from web3.logs import DISCARD
                evs = channel.events.ChannelPaid().process_receipt(
                    receipt, errors=DISCARD)
                if evs:
                    paid = int(evs[0]["args"]["amount"])
                    paid_total = int(evs[0]["args"]["paidTotal"])
            except Exception:
                log.debug("settle_channel: event decode failed", exc_info=True)
            return {"success": True, "tx_hash": tx_hex, "paid": paid,
                    "paid_total": paid_total}
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to settle channel on chain")
            return {"success": False, "error": str(e)}

    async def start_close(
        self,
        private_key: str,
        channel_id: int,
    ) -> dict[str, Any]:
        """``PaymentChannel.startClose(channelId)``, by the CLIENT: the
        provider gets one challenge window to settle, then the client
        withdraws the whole deposit (withdraw_remainder)."""
        try:
            from eth_account import Account

            w3 = self._get_web3()
            channel = self._channel_contract(w3)
            account = Account.from_key(private_key)
            chain_id = self.config.chain_id or w3.eth.chain_id
            func = channel.functions.startClose(int(channel_id))
            receipt, tx_hex = self._send_signed(
                w3, account, func, chain_id, 120_000)
            if receipt.status == 1:
                return {"success": True, "tx_hash": tx_hex}
            return {"success": False, "error": "Transaction reverted",
                    "tx_hash": tx_hex}
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to start a channel close on chain")
            return {"success": False, "error": str(e)}

    async def withdraw_remainder(
        self,
        private_key: str,
        channel_id: int,
    ) -> dict[str, Any]:
        """``PaymentChannel.withdrawRemainder(channelId)``.

        Permissionless (it's a timer): refunds the client's remainder, at
        once after the provider's close, after the challenge window after
        the client's. Anyone may trigger it; funds only
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
        except WouldRevert as e:
            return {"success": False, "error": f"not sent, it would revert: {e}",
                    "reverted": True}
        except ImportError:
            return {"success": False, "error": "web3/eth-account not installed"}
        except Exception as e:
            log.exception("Failed to withdraw remainder on chain")
            return {"success": False, "error": str(e)}

    async def get_channel(self, channel_id: int) -> dict[str, Any] | None:
        """``PaymentChannel.getChannel(channelId)`` → decoded dict.

        Returns None if the channel doesn't exist (status == 0 / None) or on
        a read failure. Fields: ``{channel_id, client, provider, token,
        deposit, claimed, closes_at, status, window}`` (``token`` lowercase
        0x, the deposit's token; amounts in its base units; status: 0 None,
        1 Open, 2 Closing, 3 Settled, 4 ClientClosing; ``claimed``: the
        cumulative paid to the provider so far, gross).
        """
        try:
            w3 = self._get_web3()
            channel = self._channel_contract(w3)
            ch = channel.functions.getChannel(int(channel_id)).call()
            client, provider, token, deposit, claimed, closes_at, status = ch[:7]
            if int(status) == 0:
                return None
            return {
                "channel_id": int(channel_id),
                "client": str(client),
                "provider": str(provider),
                "token": str(token).lower(),
                "deposit": str(deposit),
                "claimed": str(claimed),
                "closes_at": int(closes_at),
                "status": int(status),
                # The challenge window this channel opened with (governed).
                "window": int(ch[7]) if len(ch) > 7 else None,
            }
        except Exception as e:
            log.debug("get_channel(%s) failed: %s", channel_id, e)
            return None

    async def channel_payments(
        self, from_block: int, to_block: int | None = None, *,
        provider: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Every ``ChannelPaid`` log of the PaymentChannel in blocks
        ``from_block..to_block`` (``to_block`` None: the head), oldest
        first, read in chunks of ``CHANNEL_LOG_CHUNK`` blocks; only those to
        ``provider`` when given. Each is ``{channel_id, client, provider,
        token, amount, paid_total, block}`` (amounts gross, in the base
        units of ``token``, the channel's token read from the channel).

        Returns None when the logs cannot be read, so a caller that must
        agree with peers (the close) can tell a failed read from no
        payments."""
        try:
            from web3 import Web3

            w3 = self._get_web3()
            channel = self._channel_contract(w3)
            last = int(w3.eth.block_number) if to_block is None else int(to_block)
            topic = "0x" + Web3.keccak(
                text="ChannelPaid(uint256,address,address,uint256,uint256)"
            ).hex().removeprefix("0x")
            topics: list[Any] = [topic]
            if provider:
                topics += [None, None, "0x" + "00" * 12
                           + str(provider).lower().removeprefix("0x")]
            event = channel.events.ChannelPaid()
            tokens: dict[int, str] = {}
            out: list[dict[str, Any]] = []
            block = max(0, int(from_block))
            while block <= last:
                end = min(last, block + CHANNEL_LOG_CHUNK - 1)
                logs = w3.eth.get_logs({"address": channel.address,
                                        "fromBlock": block, "toBlock": end,
                                        "topics": topics})
                for lg in logs:
                    args = event.process_log(lg)["args"]
                    cid = int(args["channelId"])
                    if cid not in tokens:
                        tokens[cid] = str(channel.functions.getChannel(cid).call()[2]).lower()
                    out.append({
                        "channel_id": cid,
                        "client": Web3.to_checksum_address(args["client"]),
                        "provider": Web3.to_checksum_address(args["provider"]),
                        "token": tokens[cid],
                        "amount": int(args["amount"]),
                        "paid_total": int(args["paidTotal"]),
                        "block": int(lg["blockNumber"]),
                    })
                block = end + 1
            return out
        except Exception as e:  # noqa: BLE001
            log.debug("channel_payments(%s..%s) failed: %s",
                      from_block, to_block, e)
            return None


# ---------------------------------------------------------------------------
# Charter version (governed registry key)
# ---------------------------------------------------------------------------

def _norm_hash(value: str) -> str:
    v = (value or "").strip().lower()
    return v[2:] if v.startswith("0x") else v


def verify_charter(expected_hash: str, uri: str = "") -> dict[str, Any]:
    """Compare the local charter_hash to the jurisdiction's governed charter.

    ``expected_hash`` is the DAO Registry value of jurisdiction.<id>.charter
    (0x + sha256 hex), set at install and changed by DAO proposal. On mismatch
    the daemon logs a LOUD warning: this is a forward-only fork boundary and a
    divergent daemon would produce a different close (following the governed
    charter is future migration work; detection comes first).

    Returns ``{match, local_hash, chain_hash, uri}``; ``match`` is None when
    the jurisdiction publishes no charter.
    """
    from nodes.common.library.charter_version import charter_hash

    local = charter_hash()
    chain = _norm_hash(expected_hash)
    if not chain:
        return {"match": None, "local_hash": local, "chain_hash": None,
                "uri": uri or None}
    match = (chain == local)
    if not match:
        log.warning(
            "CHARTER DIVERGENCE: local charter_hash=%s does NOT match the "
            "jurisdiction's charter %s (%s). This daemon is running a "
            "different charter than the DAO governs; its closes will not be "
            "bit-identical to the canonical charter. Following the governed "
            "charter is future work; this is detection only.",
            local, chain, uri or "no uri",
        )
    else:
        log.info("Charter matches the jurisdiction's charter (hash=%s).", chain)
    return {"match": match, "local_hash": local, "chain_hash": chain,
            "uri": uri or None}
