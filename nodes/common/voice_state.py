"""Household voice state — chain-derived owner map + voice weights.

Spec: docs/tool_substrate.md, Decision 2026-07-08 ("ATN = money,
reputation = voice") as amended by Decision 2026-07-10 (fees-only
emission + REP-from-earnings). The federated close collapses callers to
HOUSEHOLDS (proven owner wallet, per Substrate.sol's EIP-712 owner
binding) and scales each household's damped usage/review credit by a
voice weight that is LINEAR in the household's REPUTATION:

    weight(house) = epsilon + household_reputation / reputationTotalSupply

where household_reputation = Σ reputation of every agent bound to that
owner. Linearity is the sybil property: splitting reputation across any
number of agents never gains weight; ``epsilon`` bounds what a
zero-reputation identity can carry and lets a cold-start network
(supply == 0) bootstrap.

REP SOURCE = RepToken (DAO), Decision 2026-07-10. Substrate.sol is now a
PURE MONEY contract — its reputation surface (``reputationOfAt`` /
``reputationTotalSupplyAt`` / ``agentReputation``) is DELETED. Reputation
lives DAO-side in RepToken, claimed by agents on their ratified ATN
earnings. We read each household's REP share from RepToken pinned to the
previous epoch's anchor. RepToken is TIMESTAMP-clocked (ERC20Votes,
``mode=timestamp``), so we pin on the anchor's ``timestamp`` (Anchor
struct field 8), not its block number, using the standard vote
checkpoints ``getPastVotes(addr, ts)`` / ``getPastTotalSupply(ts)``.
NOTE (judgment call, flagged): RepToken exposes no public historical
per-account BALANCE getter; the only public pinned per-account surface is
``getPastVotes``, i.e. delegated VOTING POWER. That is the correct voice
semantic — REP is the governance/voice token, and voice == voting power —
but a holder who auto-delegates to a default delegate rolls their voice
into that delegate. If per-holder rep share (independent of delegation)
is ever required, RepToken needs a public ``getPastBalance``.

OWNER-WALLET REPUTATION NOTE: there is NO separate owner-wallet term. A
bare owner wallet that never earned/claimed REP has zero votes; if the
owner ADDRESS is itself a registered agent it already appears in the
agent set and its REP is summed into its household there.

Determinism contract: every input is derived AS OF the previous epoch's
anchor (``getAnchor(anchorCount-1)`` — block number for logs, timestamp
for RepToken checkpoints, both stored on-chain at submission), so all
daemons price this epoch's voices from the identical snapshot no matter
when their refresh fires. NO ARCHIVE NODE REQUIRED: RepToken vote
checkpoints serve historical reads from current state; the agent set +
owner map derive from ``AgentRegistered`` / ``OwnerBound`` event logs up
to the snapshot block (last binding per agent wins).

The emission pool is FEES-ONLY (Decision 2026-07-10): the burned
ServiceFee shares in the snapshot anchor's window, no base floor. Zero
service volume => zero pool => zero mint. It is MONEY, ATN-denominated.

No prior anchor = no agreed snapshot: the refresh returns empty maps and
the close runs with weights=None (the uniform pre-voice behavior) —
correct for epoch 1, where nothing has minted yet anyway. No RepToken
address configured => empty rep (genesis regime), pool still fees-only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .federated_reconcile import VOICE_EPSILON

logger = logging.getLogger(__name__)

# RepToken (DAO) read surface: ERC20Votes checkpoints, timestamp-clocked.
_REP_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "account",   "type": "address"},
            {"internalType": "uint256", "name": "timepoint", "type": "uint256"},
        ],
        "name": "getPastVotes",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "timepoint", "type": "uint256"}],
        "name": "getPastTotalSupply",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

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
    rep_token_address: str = "",
    web3: Any = None,
) -> Dict[str, Any]:
    """Read {owner_map, voice_weights, rep_shares, rep_supply, supply,
    snapshot_block, emission_pool, recycled, weight_source} as of the
    previous epoch's anchor.

    ``emission_pool`` (Decision 2026-07-10, fees-only) = the burned fee
    shares (ServiceFee logs) in the snapshot anchor's window — NO base
    floor. Zero service volume => zero pool. MONEY, ATN-denominated.

    Voice weights are REPUTATION-based, read from RepToken (DAO) pinned to
    the anchor TIMESTAMP (RepToken is ERC20Votes, mode=timestamp — see
    module docstring). ``rep_supply`` is the REPUTATION total supply
    (``getPastTotalSupply``); ``weight_source`` is the literal
    ``"reputation"``. ``rep_token_address`` empty => genesis regime: no
    REP configured, empty rep maps, pool still fees-only.

    ``owner_map``: agent address -> owner wallet (lowercased 0x), only for
    agents with a bound owner at the snapshot. ``voice_weights``:
    household key -> epsilon + household_reputation/supply, rounded to 9
    decimals. Raises on RPC failure — the driver's refresh hook catches
    and keeps the previous maps.
    """
    from web3 import Web3

    w3 = web3 if web3 is not None else Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(substrate_address), abi=_VOICE_ABI,
    )

    anchor_count = contract.functions.anchorCount().call()
    if anchor_count == 0:
        # No agreed snapshot yet (epoch 1): no voices, empty fees-only pool.
        return {
            "owner_map": {},
            "voice_weights": {},
            "rep_shares": {},
            "rep_supply": 0,
            "supply": 0,
            "snapshot_block": None,
            "emission_pool": 0.0,
            "recycled": 0.0,
            "weight_source": "reputation",
        }
    anchor = contract.functions.getAnchor(anchor_count - 1).call()
    block = int(anchor[7])       # Anchor.blockNumber (struct field 7)
    timestamp = int(anchor[8])   # Anchor.timestamp (struct field 8)

    # Fees-only emission: sum the burned fee shares in this anchor's
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

    # Reputation (voice) reads: RepToken (DAO), pinned to the anchor
    # TIMESTAMP (ERC20Votes mode=timestamp). No RepToken configured =>
    # genesis regime: empty rep, close runs drift at weight 1.0.
    owner_map: Dict[str, str] = {}
    for agent in agents:
        owner = owner_of.get(agent, "")
        if owner and owner != _ZERO:
            owner_map[agent] = owner

    supply = 0
    voice_weights: Dict[str, float] = {}
    rep_shares: Dict[str, float] = {}
    if rep_token_address:
        rep = w3.eth.contract(
            address=Web3.to_checksum_address(rep_token_address), abi=_REP_ABI,
        )
        # Denominator is REPUTATION total supply (voice), not ATN.
        supply = int(rep.functions.getPastTotalSupply(timestamp).call())

        def _reputation_at(addr: str) -> int:
            # getPastVotes = delegated VOTING POWER at the pinned timestamp
            # (the only public historical per-account surface RepToken has;
            # voice == voting power — see module docstring).
            return int(rep.functions.getPastVotes(
                Web3.to_checksum_address(addr), timestamp).call())

        # household key -> raw reputation sum (int, exact — floats only at
        # the final ratio so accumulation order can't jitter the weights).
        house_rep: Dict[str, int] = {}
        for agent in agents:
            owner = owner_of.get(agent, "")
            r = _reputation_at(agent)
            if owner and owner != _ZERO:
                # Bound agent: REP rolls up to the owner household. NO
                # separate owner-wallet term — a bare owner never earns REP;
                # if it is itself a registered agent it is summed via this
                # loop. See module docstring.
                house_rep[owner] = house_rep.get(owner, 0) + r
            else:
                house_rep[agent] = house_rep.get(agent, 0) + r

        # The RAW rep share (no epsilon floor) drives drift weight;
        # voice_weights keeps the epsilon floor for mint bootstrap.
        for house in sorted(house_rep.keys()):
            share = (house_rep[house] / supply) if supply > 0 else 0.0
            voice_weights[house] = round(epsilon + share, 9)
            rep_shares[house] = round(share, 9)

    return {
        "owner_map": owner_map,
        "voice_weights": voice_weights,
        # Raw rep share per household (un-floored) + supply alias.
        "rep_shares": rep_shares,
        "rep_supply": int(supply),
        # Reputation supply — the voice-weight denominator (not ATN).
        "supply": int(supply),
        "snapshot_block": block,
        # Fees-only pool (Decision 2026-07-10): burned fees, no base floor.
        "emission_pool": round(recycled, 10),
        "recycled": round(recycled, 10),
        "weight_source": "reputation",
    }
