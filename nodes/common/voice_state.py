"""Household voice state — chain-derived owner map + voice weights.

Spec: docs/tool_substrate.md, Decision 2026-07-08 ("ATN = money,
reputation = voice"). The federated close collapses callers to
HOUSEHOLDS (proven owner wallet, per Substrate.sol's EIP-712 owner
binding) and scales each household's damped usage/review credit by a
voice weight that is LINEAR in the household's REPUTATION:

    weight(house) = epsilon + household_reputation / reputationTotalSupply

where household_reputation = Σ reputation of every agent bound to that
owner (agent mint stays on the agent address — ``recordTrainingForEpoch``
mints reputation to msg.sender — so the family's earnings count without
a sweep). Linearity is the sybil property: splitting reputation across
any number of agents never gains weight; ``epsilon`` bounds what a
zero-reputation identity can carry and lets a cold-start network
(supply == 0) bootstrap.

OWNER-WALLET REPUTATION NOTE: unlike the old balance-weighted formula,
there is NO owner-wallet term. Reputation is soulbound and minted ONLY
by ``recordTrainingForEpoch`` to the agent (msg.sender), so a plain
owner wallet — which never trains — has zero reputation. We therefore
do NOT add ``reputationOfAt(owner)`` for a bare owner wallet: it is
always zero and would only be non-zero if the owner ADDRESS is itself a
registered agent, in which case it already appears in the agent set and
its reputation is summed into its household there. Double-counting is
avoided because the owner is only added via the agent-iteration path,
never as a separate seed term. (Contrast the money view: ATN balances
CAN sit on an owner wallet, so the fleet_voice money read still counts
owner balance — but that is money, not voice.)

Determinism contract: every input is derived AS OF the previous
epoch's anchor block (``getAnchor(anchorCount-1).blockNumber``, stored
on-chain at submission), so all daemons price this epoch's voices from
the identical snapshot no matter when their refresh fires, and an agent
that mints mid-epoch (after seeing what's worth pumping) carries no
extra weight until the next epoch. NO ARCHIVE NODE REQUIRED:

  - reputation + reputation supply read through the contract's
    CHECKPOINTED endpoints (``reputationOfAt`` /
    ``reputationTotalSupplyAt`` — IVotes-style Trace208 history without
    the delegation layer), served from current state;
  - the agent set + owner map derive from ``AgentRegistered`` /
    ``OwnerBound`` event logs up to the snapshot block (last binding
    per agent wins) — logs are retained by non-archive nodes too.

The emission pool (base floor + burned ServiceFee shares) is MONEY and
stays ATN-denominated — it is untouched by the voice=reputation switch.

No prior anchor = no agreed snapshot: the refresh returns empty maps
and the close runs with weights=None (the uniform pre-voice behavior)
— correct for epoch 1, where nothing has minted yet anyway.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .federated_reconcile import BASE_EMISSION_PER_EPOCH, VOICE_EPSILON

logger = logging.getLogger(__name__)

_VOICE_ABI = [
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
    {
        "inputs": [
            {"internalType": "address", "name": "agent", "type": "address"},
            {"internalType": "uint256", "name": "blockNumber", "type": "uint256"},
        ],
        "name": "reputationOfAt",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "blockNumber", "type": "uint256"}],
        "name": "reputationTotalSupplyAt",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
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
            {"indexed": True,  "internalType": "address", "name": "agent",     "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "owner",     "type": "address"},
            {"indexed": True,  "internalType": "address", "name": "parent",    "type": "address"},
            {"indexed": False, "internalType": "address", "name": "prevOwner", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "nonce",     "type": "uint256"},
        ],
        "name": "OwnerBound",
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
]

# Chain amounts are ATN x 1e6 (the agent-side submitter scaling);
# emission pools are float ATN units on the close side.
_ATN_SCALE = 1_000_000.0

_ZERO = "0x0000000000000000000000000000000000000000"


# (rpc_url-ish key, address) -> deployment block. Found once per
# process; the contract's creation block never changes.
_DEPLOY_BLOCK_CACHE: Dict[Any, int] = {}


def _deployment_block(w3, address: str) -> int:
    """Lowest block at which the contract has code (binary search on
    eth_getCode, ~log2(head) RPC calls, cached). Event scans start
    here instead of block 0 — public RPCs cap eth_getLogs ranges."""
    key = (id(w3.provider), address.lower())
    hit = _DEPLOY_BLOCK_CACHE.get(key)
    if hit is not None:
        return hit
    lo, hi = 0, w3.eth.block_number
    if not w3.eth.get_code(address, block_identifier=hi):
        return hi  # no code at head — nothing to scan anyway
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            has_code = bool(w3.eth.get_code(address, block_identifier=mid))
        except Exception:
            # Non-archive node can't serve old getCode — assume no code
            # there and keep moving up; worst case we start the scan a
            # little late, never early.
            has_code = False
        if has_code:
            hi = mid
        else:
            lo = mid + 1
    _DEPLOY_BLOCK_CACHE[key] = lo
    return lo


def _get_logs_range(event, from_block: int, to_block: int):
    try:
        return event.get_logs(from_block=from_block, to_block=to_block)
    except TypeError:
        return event.get_logs(fromBlock=from_block, toBlock=to_block)


def _get_logs(event, w3, address: str, to_block: int):
    """Fetch an event's logs [deployment, to_block], splitting the
    range adaptively when the RPC rejects it as too large (public
    endpoints cap eth_getLogs spans)."""
    start = _deployment_block(w3, address)
    if start > to_block:
        return []
    spans = [(start, to_block)]
    out = []
    while spans:
        lo, hi = spans.pop()
        try:
            out.extend(_get_logs_range(event, lo, hi))
        except Exception as e:
            if lo >= hi or "range" not in str(e).lower():
                raise
            mid = (lo + hi) // 2
            spans.append((mid + 1, hi))
            spans.append((lo, mid))
    return out


def read_voice_state(
    substrate_address: str,
    rpc_url: str = "",
    *,
    epsilon: float = VOICE_EPSILON,
    web3: Any = None,
) -> Dict[str, Any]:
    """Read {owner_map, voice_weights, supply, snapshot_block,
    emission_pool, recycled, weight_source} from Substrate.sol as of the
    previous epoch's anchor block. ``emission_pool`` =
    BASE_EMISSION_PER_EPOCH + the burned fee shares (ServiceFee logs) in
    the snapshot anchor's window — the fee-recycled emission input to the
    close. The emission pool is MONEY and stays ATN-denominated.

    Voice weights are REPUTATION-based (ratified 2026-07-08: ATN = money,
    reputation = voice). ``supply`` is the REPUTATION total supply used
    as the weight denominator (NOT the ATN supply); ``weight_source`` is
    the literal string ``"reputation"`` so any consumer that historically
    read ``supply`` as ATN can detect the switch.

    ``owner_map``: agent address -> owner wallet (lowercased 0x), only
    for agents with a bound owner at the snapshot. ``voice_weights``:
    household key -> epsilon + household_reputation/supply, rounded to 9
    decimals; keys are owner wallets for bound fleets and agent
    addresses for unbound agents (matching the close's household
    fallback). Raises on RPC failure — the driver's refresh hook
    catches and keeps the previous maps.
    """
    from web3 import Web3

    w3 = web3 if web3 is not None else Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(substrate_address), abi=_VOICE_ABI,
    )

    anchor_count = contract.functions.anchorCount().call()
    if anchor_count == 0:
        # No agreed snapshot yet (epoch 1): no voices, floor-only pool.
        return {
            "owner_map": {},
            "voice_weights": {},
            "supply": 0,
            "snapshot_block": None,
            "emission_pool": BASE_EMISSION_PER_EPOCH,
            "recycled": 0.0,
            "weight_source": "reputation",
        }
    anchor = contract.functions.getAnchor(anchor_count - 1).call()
    block = int(anchor[7])  # Anchor.blockNumber (struct field 7)

    # Fee-recycled emission: sum the burned fee shares in this anchor's
    # window — (previous anchor block, this anchor block]. Every fee
    # lands in exactly one window, so recycling conserves across epochs.
    if anchor_count >= 2:
        prev_anchor = contract.functions.getAnchor(anchor_count - 2).call()
        window_start = int(prev_anchor[7]) + 1
    else:
        window_start = _deployment_block(w3, contract.address)
    burned_raw = 0
    for log in _get_logs(contract.events.ServiceFee, w3,
                         contract.address, block):
        if int(log["blockNumber"]) >= window_start:
            burned_raw += int(log["args"]["burned"])
    recycled = burned_raw / _ATN_SCALE

    # Agent set + owner map from event logs up to the snapshot block —
    # deterministic (chain history) and served by non-archive nodes.
    # Last OwnerBound per agent wins (rotation support), ordered by
    # (blockNumber, logIndex).
    agents = sorted({
        str(log["args"]["agent"]).lower()
        for log in _get_logs(
            contract.events.AgentRegistered, w3, contract.address, block)
    })
    bindings = sorted(
        _get_logs(contract.events.OwnerBound, w3, contract.address, block),
        key=lambda log: (int(log["blockNumber"]), int(log["logIndex"])),
    )
    owner_of: Dict[str, str] = {}
    for log in bindings:
        owner_of[str(log["args"]["agent"]).lower()] = (
            str(log["args"]["owner"]).lower())

    # Denominator is REPUTATION supply (voice), not ATN supply (money).
    supply = int(contract.functions.reputationTotalSupplyAt(block).call())

    def _reputation_at(addr: str) -> int:
        return int(contract.functions.reputationOfAt(
            Web3.to_checksum_address(addr), block).call())

    owner_map: Dict[str, str] = {}
    # household key -> raw reputation sum (int, exact — floats only at the
    # final ratio so accumulation order can't jitter the weights).
    house_rep: Dict[str, int] = {}
    for agent in agents:
        owner = owner_of.get(agent, "")
        rep = _reputation_at(agent)
        if owner and owner != _ZERO:
            # Bound agent: reputation rolls up to the owner household.
            # NO separate owner-wallet reputation term — a bare owner
            # wallet never trains, so reputationOfAt(owner) is always
            # zero. If the owner address is itself a registered agent it
            # already appears in ``agents`` and is summed via this loop
            # (its household is then its own owner binding); adding it as
            # a seed here would double-count. See module docstring.
            owner_map[agent] = owner
            house_rep[owner] = house_rep.get(owner, 0) + rep
        else:
            # Unbound agent: its own household (matches _household()).
            house_rep[agent] = house_rep.get(agent, 0) + rep

    voice_weights: Dict[str, float] = {}
    for house in sorted(house_rep.keys()):
        share = (house_rep[house] / supply) if supply > 0 else 0.0
        voice_weights[house] = round(epsilon + share, 9)

    return {
        "owner_map": owner_map,
        "voice_weights": voice_weights,
        # Reputation supply — the voice-weight denominator (not ATN).
        "supply": int(supply),
        "snapshot_block": block,
        "emission_pool": round(BASE_EMISSION_PER_EPOCH + recycled, 10),
        "recycled": round(recycled, 10),
        "weight_source": "reputation",
    }
