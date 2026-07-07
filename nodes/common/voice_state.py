"""Household voice state — chain-derived owner map + voice weights.

Spec: docs/tool_substrate.md, Decision 2026-07-08 addendum
(balance-weighted voice). The federated close collapses callers to
HOUSEHOLDS (proven owner wallet, per Substrate.sol's EIP-712 owner
binding) and scales each household's damped usage/review credit by a
voice weight that is LINEAR in the household's ATN:

    weight(house) = epsilon + household_ATN / atnTotalSupply

where household_ATN = owner wallet balance + Σ balances of every agent
bound to that owner (agent mint stays on the agent address —
``recordTrainingForEpoch`` mints to msg.sender — so the family's
earnings count without a sweep). Linearity is the sybil property:
splitting a balance across any number of wallets or agents never gains
weight; ``epsilon`` bounds what a zero-balance identity can carry and
lets a cold-start network (supply == 0) bootstrap.

Determinism contract: STRONGER than a "same chain state" hope — every
read is PINNED to the previous epoch's anchor block
(``getAnchor(anchorCount-1).blockNumber``, stored on-chain at
submission). All daemons therefore price this epoch's voices from the
identical snapshot no matter when their refresh fires, and a wallet
funded mid-epoch (after seeing what's worth pumping) carries no weight
until the next epoch. No prior anchor = no agreed snapshot: the
refresh returns empty maps and the close runs with weights=None (the
uniform pre-voice behavior) — correct for epoch 1, where nothing has
minted yet anyway.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .federated_reconcile import VOICE_EPSILON

logger = logging.getLogger(__name__)

_VOICE_ABI = [
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
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "agentOwner",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
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
        "inputs": [],
        "name": "atnTotalSupply",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "anchorCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "name": "getAnchor",
        "outputs": [{
            "components": [
                {"internalType": "string",  "name": "epochId",       "type": "string"},
                {"internalType": "bytes32", "name": "epochRoot",     "type": "bytes32"},
                {"internalType": "bytes32", "name": "prevEpochRoot", "type": "bytes32"},
                {"internalType": "bytes32", "name": "prevAnchorHash","type": "bytes32"},
                {"internalType": "string",  "name": "agentMintCid",  "type": "string"},
                {"internalType": "bytes32", "name": "payloadHash",   "type": "bytes32"},
                {"internalType": "address", "name": "submitter",     "type": "address"},
                {"internalType": "uint256", "name": "blockNumber",   "type": "uint256"},
                {"internalType": "uint256", "name": "timestamp",     "type": "uint256"},
                {"internalType": "bytes32", "name": "agentMintRoot", "type": "bytes32"},
            ],
            "internalType": "struct Substrate.Anchor",
            "name": "",
            "type": "tuple",
        }],
        "stateMutability": "view",
        "type": "function",
    },
]

_ZERO = "0x0000000000000000000000000000000000000000"


def read_voice_state(
    substrate_address: str,
    rpc_url: str,
    *,
    epsilon: float = VOICE_EPSILON,
) -> Dict[str, Any]:
    """Read {owner_map, voice_weights, supply, snapshot_block} from
    Substrate.sol, PINNED to the previous epoch's anchor block.

    The snapshot block is ``getAnchor(anchorCount-1).blockNumber`` —
    on-chain, agreed, and pre-dating this epoch — so every daemon
    derives the identical maps regardless of when its refresh fires.
    With no anchor yet (epoch 1) there is no agreed snapshot: returns
    empty maps (close runs weights=None, the uniform behavior).

    ``owner_map``: agent address -> owner wallet (lowercased 0x), only
    for agents with a bound owner. ``voice_weights``: household key ->
    epsilon + household_ATN/supply, rounded to 9 decimals; keys are
    owner wallets for bound fleets and agent addresses for unbound
    agents (matching the close's household fallback). Raises on RPC
    failure — the driver's refresh hook catches and keeps the previous
    maps.
    """
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(substrate_address), abi=_VOICE_ABI,
    )

    anchor_count = contract.functions.anchorCount().call()
    if anchor_count == 0:
        return {
            "owner_map": {},
            "voice_weights": {},
            "supply": 0,
            "snapshot_block": None,
        }
    anchor = contract.functions.getAnchor(anchor_count - 1).call()
    block = int(anchor[7])  # Anchor.blockNumber (struct field 7)

    def _call(fn):
        return fn.call(block_identifier=block)

    count = _call(contract.functions.registeredAgentCount())
    supply = _call(contract.functions.atnTotalSupply())

    owner_map: Dict[str, str] = {}
    # household key -> raw balance sum (int, exact — floats only at the
    # final ratio so accumulation order can't jitter the weights).
    house_balance: Dict[str, int] = {}
    seen_owners: set = set()
    for i in range(count):
        agent = _call(contract.functions.getRegisteredAgent(i))
        agent_lc = str(agent).lower()
        owner = _call(contract.functions.agentOwner(agent))
        owner_lc = str(owner).lower()
        bal = int(_call(contract.functions.balanceOf(agent)))
        if owner_lc and owner_lc != _ZERO:
            owner_map[agent_lc] = owner_lc
            house_balance[owner_lc] = house_balance.get(owner_lc, 0) + bal
            if owner_lc not in seen_owners:
                seen_owners.add(owner_lc)
                house_balance[owner_lc] += int(
                    _call(contract.functions.balanceOf(owner)))
        else:
            # Unbound agent: its own household (matches _household()).
            house_balance[agent_lc] = house_balance.get(agent_lc, 0) + bal

    voice_weights: Dict[str, float] = {}
    for house in sorted(house_balance.keys()):
        share = (house_balance[house] / supply) if supply > 0 else 0.0
        voice_weights[house] = round(epsilon + share, 9)

    return {
        "owner_map": owner_map,
        "voice_weights": voice_weights,
        "supply": int(supply),
        "snapshot_block": block,
    }
