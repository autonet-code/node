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

Determinism contract: same as ``agent_owner_map`` — every daemon
reading the same chain state derives the same maps. Reads happen at
the driver's refresh hook just before the close; a lagging chain view
is the accepted risk (same class as the owner map), with the
anchored-block-pinned read as the named upgrade.
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
]

_ZERO = "0x0000000000000000000000000000000000000000"


def read_voice_state(
    substrate_address: str,
    rpc_url: str,
    *,
    epsilon: float = VOICE_EPSILON,
) -> Dict[str, Any]:
    """Read {owner_map, voice_weights, supply} from Substrate.sol.

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

    count = contract.functions.registeredAgentCount().call()
    supply = contract.functions.atnTotalSupply().call()

    owner_map: Dict[str, str] = {}
    # household key -> raw balance sum (int, exact — floats only at the
    # final ratio so accumulation order can't jitter the weights).
    house_balance: Dict[str, int] = {}
    seen_owners: set = set()
    for i in range(count):
        agent = contract.functions.getRegisteredAgent(i).call()
        agent_lc = str(agent).lower()
        owner = contract.functions.agentOwner(agent).call()
        owner_lc = str(owner).lower()
        bal = int(contract.functions.balanceOf(agent).call())
        if owner_lc and owner_lc != _ZERO:
            owner_map[agent_lc] = owner_lc
            house_balance[owner_lc] = house_balance.get(owner_lc, 0) + bal
            if owner_lc not in seen_owners:
                seen_owners.add(owner_lc)
                house_balance[owner_lc] += int(
                    contract.functions.balanceOf(owner).call())
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
    }
