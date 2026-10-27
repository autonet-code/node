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
struct field 8), not its block number, using RepToken's balance
checkpoints ``getPastBalance(addr, ts)`` / ``getPastTotalSupply(ts)``.
BALANCES, not votes (Decision 2026-10-26): the substrate weighs what a
household holds, so delegation (a governance act) never moves substrate
weight. An agent delegating out keeps its household's weight; an outsider
delegating in adds none. Governor votes and the anchor election keep
reading delegated ``getVotes``.

OWNER-WALLET REPUTATION NOTE (Decision 2026-10-27): the owner wallet is
the household's claimant, so it holds the household's REP. A household's
REP is the owner's own balance plus its bound agents' balances (agents
may still hold REP from earlier claims). An owner address that is itself
a registered agent is counted once, through the agent set.

Determinism contract: every input is derived AS OF the previous epoch's
anchor (``getAnchor(anchorCount-1)`` — block number for logs, timestamp
for RepToken checkpoints, both stored on-chain at submission), so all
daemons price this epoch's voices from the identical snapshot no matter
when their refresh fires. NO ARCHIVE NODE REQUIRED: RepToken vote
checkpoints serve historical reads from current state; the agent set +
owner map derive from ``AgentRegistered`` / ``OwnerBound`` event logs up
to the snapshot block (last binding per agent wins).

The emission pool is FEES-ONLY (Decision 2026-07-10): the burned
ServiceFee shares in the snapshot anchor's window, no base floor. Since
2026-10-12 it is read from the anchor struct's ``windowBurned`` (the exact
quantity Substrate caps the next epoch's mints at), not summed from logs. Zero
service volume => zero pool => zero mint. It is MONEY, ATN-denominated.

No prior anchor = no agreed snapshot: the refresh returns empty maps and
the close runs with weights=None (the uniform pre-voice behavior) —
correct for epoch 1, where nothing has minted yet anyway. No RepToken
address configured => empty rep (genesis regime), pool still fees-only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .federated_reconcile import VOICE_EPSILON

from .containment import params_from_writes

logger = logging.getLogger(__name__)

# RepToken (DAO) read surface: ERC20Votes checkpoints, timestamp-clocked.
_REP_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "account",   "type": "address"},
            {"internalType": "uint256", "name": "timepoint", "type": "uint256"},
        ],
        "name": "getPastBalance",
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
        "inputs": [{"internalType": "uint256", "name": "blockNumber", "type": "uint256"}],
        "name": "atnTotalSupplyAt",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "fromTs", "type": "uint256"},
                   {"internalType": "uint256", "name": "toTs", "type": "uint256"}],
        "name": "burnedBetween",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "deployedAt",
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
                {"internalType": "uint256", "name": "windowBurned",  "type": "uint256"},
                {"internalType": "uint256", "name": "mintPool",      "type": "uint256"},
                {"internalType": "uint256", "name": "minted",        "type": "uint256"},
                {"internalType": "uint256", "name": "cutoff",        "type": "uint256"},
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
    here instead of block 0 — public RPCs cap eth_getLogs ranges.

    Only used when no deployment block is configured. A failed read
    raises (Decision 2026-10-26): a non-archive node that cannot serve old
    state used to count as "no code yet", which started the scan late and
    silently dropped early events, so the close disagreed with peers."""
    key = (id(w3.provider), address.lower())
    hit = _DEPLOY_BLOCK_CACHE.get(key)
    if hit is not None:
        return hit
    lo, hi = 0, w3.eth.block_number
    if not w3.eth.get_code(address, block_identifier=hi):
        return hi  # no code at head — nothing to scan anyway
    while lo < hi:
        mid = (lo + hi) // 2
        has_code = bool(w3.eth.get_code(address, block_identifier=mid))
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


def _get_logs(event, w3, address: str, to_block: int,
              start_block: Optional[int] = None):
    """Fetch an event's logs [deployment, to_block], splitting the
    range adaptively when the RPC rejects it as too large (public
    endpoints cap eth_getLogs spans). ``start_block`` is the configured
    deployment block of the jurisdiction; without it the start is found
    by ``_deployment_block``."""
    start = (int(start_block) if start_block is not None
             else _deployment_block(w3, address))
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


# ---------------------------------------------------------------------------
# ToolBond review snapshot (docs/multi_jurisdiction.md decision 10).
#
# ToolBond (trustless-contracts 448023a) is the single source of tool
# ratings: staked reviews, committed and revealed per agent per epoch, settle
# per tool round by REP weight. The close reads, pinned to the SAME snapshot
# block as the voice weights:
#
#   - each tool's rating per charter axis and its harm over every round
#     finalized at or before the block (what toolRating / toolHarm return at
#     that block), and
#   - the rounds finalized in the window since the anchor before it, with
#     each review's stake, accuracy and the axes it scored, for the reviewer
#     share of the tool's mint.
#
# An eth_call at the anchor block needs archive state, so the state is
# REBUILT from ToolBond's logs up to the snapshot block, like the owner map:
# a round's per-axis sums and masses are the sums of its Reviewed events
# (the contract's _record writes exactly these), and a round joins the
# tool's rating when its Finalized event lands. Two facts the logs do not
# carry are read from immutable views at the latest block: an epoch's axis
# ids and REP supply (epochInfo, fixed once the epoch starts, which precedes
# any reveal) and a finalized review's accuracy (reviewAt, fixed once the
# round is Done). Any daemon, archive or not, derives the same snapshot.
# ---------------------------------------------------------------------------

REVIEWER_SHARE_BPS_DEFAULT = 4000
_NO_SCORE = -128
_NO_HARM = 255
_UNINFORMATIVE = 2 ** 64 - 1


def _ev(name: str, fields: list) -> Dict[str, Any]:
    return {
        "anonymous": False, "name": name, "type": "event",
        "inputs": [{"indexed": idx, "internalType": t, "name": n, "type": t}
                   for n, t, idx in fields],
    }


_TOOL_BOND_EVENTS_ABI = [
    _ev("Reviewed", [("digest", "bytes32", True), ("epoch", "uint256", True),
                     ("reviewer", "address", True), ("scores", "bytes", False),
                     ("harm", "uint8", False), ("stake", "uint256", False),
                     ("weight", "uint256", False), ("evidenceHash", "bytes32", False)]),
    _ev("Finalized", [("digest", "bytes32", True), ("epoch", "uint256", True),
                      ("harm", "uint256", False), ("bondSlashed", "uint256", False),
                      ("lossPool", "uint256", False), ("repPool", "uint256", False)]),
    _ev("Overruled", [("digest", "bytes32", True), ("epoch", "uint256", True),
                      ("harm", "uint256", False)]),
    _ev("HarmSettled", [("digest", "bytes32", True), ("epoch", "uint256", True),
                        ("harm", "uint256", False), ("appealEnds", "uint48", False)]),
    # Added in trustless-contracts baab631; older deploys simply emit none
    # and the views below fill in.
    _ev("ReviewSettled", [("digest", "bytes32", True), ("epoch", "uint256", True),
                          ("reviewer", "address", True), ("accuracy", "uint256", False),
                          ("harmAccuracy", "uint256", False)]),
    _ev("Bonded", [("digest", "bytes32", True), ("author", "address", True),
                    ("amount", "uint256", False), ("stake", "uint256", False)]),
    _ev("UnbondRequested", [("digest", "bytes32", True), ("availableAt", "uint256", False)]),
    _ev("Unbonded", [("digest", "bytes32", True), ("author", "address", True),
                     ("amount", "uint256", False)]),
    _ev("EpochStarted", [("epoch", "uint256", True), ("charterHash", "bytes32", False),
                         ("axisIds", "bytes32[]", False), ("snapshot", "uint48", False),
                         ("supply", "uint256", False)]),
    _ev("ParamsSet", [("minStake", "uint256", False), ("revealWindow", "uint256", False),
                      ("degradeBps", "uint256", False), ("authorSlashBps", "uint256", False),
                      ("unbondDelay", "uint256", False), ("reviewerShareBps", "uint256", False),
                      ("harmWeightBps", "uint256", False),
                      ("harmQuorumBps", "uint256", False)]),
]

_REVIEW_COMPONENTS = [
    {"name": "reviewer", "type": "address"}, {"name": "harm", "type": "uint8"},
    {"name": "claimed", "type": "bool"}, {"name": "accuracy", "type": "uint64"},
    {"name": "scores", "type": "bytes16"}, {"name": "stake", "type": "uint128"},
    {"name": "weight", "type": "uint128"}, {"name": "harmAccuracy", "type": "uint64"},
]

_TOOL_BOND_VIEWS_ABI = [
    {"name": "epochInfo", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "epoch", "type": "uint256"}],
     "outputs": [{"name": "started", "type": "bool"}, {"name": "charterHash", "type": "bytes32"},
                 {"name": "axisIds", "type": "bytes32[]"}, {"name": "snapshot", "type": "uint48"},
                 {"name": "supply", "type": "uint256"}, {"name": "revealDeadline", "type": "uint256"}]},
    {"name": "reviewAt", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "digest", "type": "bytes32"}, {"name": "epoch", "type": "uint256"},
                {"name": "i", "type": "uint256"}],
     "outputs": [{"name": "", "type": "tuple", "components": _REVIEW_COMPONENTS}]},
]


def _hex32(value: Any) -> str:
    raw = value.hex() if hasattr(value, "hex") else str(value)
    return str(raw).lower().removeprefix("0x")


def _int8(b: int) -> int:
    return b - 256 if b > 127 else b


def tool_review_snapshot_from_logs(
    records: List[Dict[str, Any]],
    *,
    block: int,
    since_block: int,
    epochs: Dict[int, Dict[str, Any]],
    accuracies: Dict[Tuple[str, int, int], Optional[int]],
    axis_ids: List[str],
    default_bps: int = REVIEWER_SHARE_BPS_DEFAULT,
    excluded_reviewers: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Fold ToolBond records into the close's review snapshot. Pure, so it is
    testable without a chain.

    ``excluded_reviewers``: reviewer addresses whose reviews move no rating
    (sanctioned households, docs/accountability.md). Their reviews still
    appear in ``rounds``, so the reviewer-share split is unchanged and their
    share is withheld at the mint instead of passing to other reviewers.

    ``records``: ``{"block", "index", "event", "args"}`` dicts (any order;
    folded in (block, logIndex) order). ``epochs``: epoch -> {"axis_ids":
    [hex keccak of each charter axis id, in that epoch's order], "supply":
    int}. ``accuracies``: (digest, epoch, review index) -> accuracy (1e18
    scale, None when uninformative), needed only for rounds finalized in
    (since_block, block]. ``axis_ids``: the active charter's axis ids; a
    ToolBond axis key outside it is dropped (a removed axis).

    Returns::

      {"reviewer_share_bps": int,
       "tools": {digest: {"axes": {axis_id: {"sum": int, "mass": int}},
                          "harm": {"sum": int, "mass": int}}},
       "rounds": {digest: [{"epoch", "supply", "prior_mass": {axis_id|"harm": int},
                            "reviews": [{"reviewer", "stake", "accuracy",
                                         "axes": [axis ids scored], "harm": bool}]}]}}
    """
    from eth_utils import keccak

    key_to_axis = {keccak(text=a).hex().lower().removeprefix("0x"): a for a in axis_ids}
    ordered = sorted(records, key=lambda r: (int(r["block"]), int(r["index"])))
    bps: Optional[int] = None
    reviews: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    finalized: List[Tuple[str, int, int]] = []   # (digest, epoch, block), in order
    overruled: Dict[Tuple[str, int], int] = {}    # DAO's harm 0..100 for a round
    cooling: List[Tuple[str, int]] = []           # harm settled, appeal window open
    bonds: Dict[str, Dict[str, Any]] = {}         # digest -> {author, stake, unbonding}
    min_stake = 0
    settled: Dict[Tuple[str, int, str], int] = {}  # (digest, epoch, reviewer) -> accuracy
    epochs = dict(epochs)
    for r in ordered:
        if int(r["block"]) > int(block):
            break
        name, a = r["event"], r["args"]
        if name == "ParamsSet":
            bps = int(a["reviewerShareBps"])
            min_stake = int(a["minStake"])
        elif name == "Bonded":
            b = bonds.setdefault(_hex32(a["digest"]), {"author": "", "stake": 0,
                                                       "unbonding": False})
            b["author"] = str(a["author"]).lower()
            b["stake"] = int(a["stake"])
            b["unbonding"] = False   # a top-up cancels a pending unbond
        elif name == "UnbondRequested":
            if _hex32(a["digest"]) in bonds:
                bonds[_hex32(a["digest"])]["unbonding"] = True
        elif name == "Unbonded":
            bonds.pop(_hex32(a["digest"]), None)
        elif name == "Reviewed":
            reviews.setdefault((_hex32(a["digest"]), int(a["epoch"])), []).append({
                "reviewer": str(a["reviewer"]).lower(),
                "scores": bytes(a["scores"]), "harm": int(a["harm"]),
                "stake": int(a["stake"]), "weight": int(a["weight"]),
            })
        elif name == "Finalized":
            finalized.append((_hex32(a["digest"]), int(a["epoch"]), int(r["block"])))
            b = bonds.get(_hex32(a["digest"]))
            if b is not None:
                b["stake"] = max(0, b["stake"] - int(a["bondSlashed"]))
        elif name == "Overruled":
            overruled[(_hex32(a["digest"]), int(a["epoch"]))] = int(a["harm"])
        elif name == "HarmSettled":
            cooling.append((_hex32(a["digest"]), int(a["epoch"])))
        elif name == "ReviewSettled":
            settled[(_hex32(a["digest"]), int(a["epoch"]),
                     str(a["reviewer"]).lower())] = int(a["accuracy"])
        elif name == "EpochStarted":
            epochs.setdefault(int(a["epoch"]), {
                "axis_ids": [_hex32(x) for x in a["axisIds"]],
                "supply": int(a["supply"])})

    excluded = {str(a).lower() for a in (excluded_reviewers or ())}
    tools: Dict[str, Dict[str, Any]] = {}
    rounds: Dict[str, List[Dict[str, Any]]] = {}
    for digest, epoch, fin_block in finalized:
        revs = reviews.get((digest, epoch), [])
        info = epochs.get(epoch) or {}
        keys = [str(k).lower().removeprefix("0x") for k in (info.get("axis_ids") or [])]
        sums: Dict[str, List[int]] = {}
        h_sum = h_mass = 0
        for v in revs:
            w = 0 if v["reviewer"] in excluded else v["weight"]
            for i, b in enumerate(v["scores"][:len(keys)]):
                s = _int8(b)
                axis = key_to_axis.get(keys[i])
                if s == _NO_SCORE or w == 0 or axis is None:
                    continue
                cell = sums.setdefault(axis, [0, 0])
                cell[0] += w * s
                cell[1] += w
            if v["harm"] != _NO_HARM and w > 0:
                h_sum += w * v["harm"]
                h_mass += w
        if (digest, epoch) in overruled:
            # The DAO's ruling replaces the round's harm, at the same REP
            # mass (as ToolBond.toolHarm counts it since baab631).
            h_sum = h_mass * overruled[(digest, epoch)]
        tool = tools.setdefault(digest, {"axes": {}, "harm": {"sum": 0, "mass": 0}})
        prior = {a: int(c["mass"]) for a, c in tool["axes"].items()}
        prior["harm"] = int(tool["harm"]["mass"])
        if fin_block > int(since_block):
            entries = []
            for i, v in enumerate(revs):
                scored = [key_to_axis[keys[j]] for j, b in enumerate(v["scores"][:len(keys)])
                          if _int8(b) != _NO_SCORE and keys[j] in key_to_axis]
                acc = settled.get((digest, epoch, v["reviewer"]),
                                  accuracies.get((digest, epoch, i)))
                entries.append({
                    "reviewer": v["reviewer"], "stake": v["stake"],
                    "accuracy": None if acc is None or int(acc) == _UNINFORMATIVE else int(acc),
                    "axes": sorted(scored), "harm": v["harm"] != _NO_HARM,
                })
            rounds.setdefault(digest, []).append({
                "epoch": epoch, "supply": int(info.get("supply") or 0),
                "prior_mass": dict(sorted(prior.items())), "reviews": entries,
            })
        for axis, (sm, ms) in sums.items():
            cell = tool["axes"].setdefault(axis, {"sum": 0, "mass": 0})
            cell["sum"] += sm
            cell["mass"] += ms
        tool["harm"]["sum"] += h_sum
        tool["harm"]["mass"] += h_mass

    # Harm settled but still in its appeal window (Decision 2026-10-27):
    # containment acts on it at once, since containment is reversible (an
    # overrule replaces it); ratings, mint and slashes wait for Finalized.
    # Kept out of "tools", so the close's inputs do not change.
    done = {(d, e) for d, e, _b in finalized}
    cooling_harm: Dict[str, Dict[str, int]] = {}
    for digest, epoch in cooling:
        if (digest, epoch) in done:
            continue
        h_sum = h_mass = 0
        for v in reviews.get((digest, epoch), []):
            w = 0 if v["reviewer"] in excluded else v["weight"]
            if v["harm"] != _NO_HARM and w > 0:
                h_sum += w * v["harm"]
                h_mass += w
        if (digest, epoch) in overruled:
            h_sum = h_mass * overruled[(digest, epoch)]
        cell = cooling_harm.setdefault(digest, {"sum": 0, "mass": 0})
        cell["sum"] += h_sum
        cell["mass"] += h_mass

    return {
        "reviewer_share_bps": int(default_bps if bps is None else bps),
        "tools": {d: {"axes": dict(sorted(t["axes"].items())), "harm": dict(t["harm"])}
                  for d, t in sorted(tools.items())},
        "rounds": {d: rs for d, rs in sorted(rounds.items())},
        "cooling_harm": dict(sorted(cooling_harm.items())),
        # What may mint (Decision 2026-10-27): tools bonded at the block.
        "bonds": dict(sorted(bonds.items())),
        "min_stake": int(min_stake),
    }


def read_tool_review_snapshot(
    w3: Any, tool_bond_address: str, block: int, *, since_block: int = 0,
    axis_ids: Optional[List[str]] = None,
    excluded_reviewers: Optional[Iterable[str]] = None,
    start_block: Optional[int] = None,
) -> Dict[str, Any]:
    """ToolBond review snapshot as of ``block`` (the snapshot anchor's
    block), with the reviewer-share rounds finalized after ``since_block``
    (the anchor before it). Raises on RPC failure, like the rest of the
    voice read: the driver keeps its previous snapshot."""
    from web3 import Web3

    if axis_ids is None:
        from .world_model_substrate.charter import active_charter
        axis_ids = list(active_charter().ids)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(tool_bond_address),
        abi=_TOOL_BOND_EVENTS_ABI + _TOOL_BOND_VIEWS_ABI,
    )
    records: List[Dict[str, Any]] = []
    for spec in _TOOL_BOND_EVENTS_ABI:
        event = getattr(contract.events, spec["name"])
        for log in _get_logs(event, w3, contract.address, block, start_block):
            records.append({
                "block": int(log["blockNumber"]),
                "index": int(log["logIndex"]),
                "event": spec["name"],
                "args": dict(log["args"]),
            })
    epochs: Dict[int, Dict[str, Any]] = {}
    review_counts: Dict[Tuple[str, int], int] = {}
    window: List[Tuple[str, int]] = []
    for r in records:
        if r["event"] == "Reviewed":
            k = (_hex32(r["args"]["digest"]), int(r["args"]["epoch"]))
            review_counts[k] = review_counts.get(k, 0) + 1
        elif r["event"] == "Finalized":
            e = int(r["args"]["epoch"])
            if e not in epochs and not any(
                    x["event"] == "EpochStarted" and int(x["args"]["epoch"]) == e
                    for x in records):
                info = contract.functions.epochInfo(e).call()
                epochs[e] = {"axis_ids": [_hex32(x) for x in info[2]],
                             "supply": int(info[4])}
            if int(r["block"]) > int(since_block):
                window.append((_hex32(r["args"]["digest"]), e))
    accuracies: Dict[Tuple[str, int, int], Optional[int]] = {}
    settled_keys = {(_hex32(x["args"]["digest"]), int(x["args"]["epoch"]))
                    for x in records if x["event"] == "ReviewSettled"}
    for digest, e in sorted(set(window) - settled_keys):
        for i in range(review_counts.get((digest, e), 0)):
            v = contract.functions.reviewAt(bytes.fromhex(digest), e, i).call()
            accuracies[(digest, e, i)] = int(v[3])
    return tool_review_snapshot_from_logs(
        records, block=block, since_block=since_block, epochs=epochs,
        accuracies=accuracies, axis_ids=axis_ids,
        excluded_reviewers=excluded_reviewers)


# ---------------------------------------------------------------------------
# Pinned charter (docs/multi_jurisdiction.md decision 5).
#
# The DAO Registry holds the charter text under jurisdiction.autonet.charter
# and emits ``RegistryUpdated(string key, string value)`` (both fields
# non-indexed) on every write. The charter in force for an epoch is the last
# value written at or before the snapshot anchor's block, rebuilt from those
# logs like the owner map (non-archive nodes serve them).
# ---------------------------------------------------------------------------

CHARTER_REGISTRY_KEY = "jurisdiction.autonet.charter"

_REGISTRY_EVENTS_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "string", "name": "key", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "value", "type": "string"},
        ],
        "name": "RegistryUpdated",
        "type": "event",
    },
]


def charter_text_from_logs(
    records: List[Dict[str, Any]], block: int,
    key: str = CHARTER_REGISTRY_KEY,
) -> Optional[str]:
    """Last ``value`` written for ``key`` at or before ``block``.

    ``records`` are ``{"block", "index", "key", "value"}``; ordered by
    (block, logIndex). Returns None when the key was never written, or when
    the last value is not charter text (a legacy 0x hash)."""
    last: Optional[str] = None
    for rec in sorted(records, key=lambda r: (int(r["block"]), int(r["index"]))):
        if int(rec["block"]) > int(block):
            break
        if str(rec.get("key")) == key:
            last = str(rec.get("value") or "")
    if last is None or not last.lstrip().startswith("{"):
        return None
    return last


def _registry_records(
    w3: Any, dao_registry_address: str, block: int,
    start_block: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Every RegistryUpdated write up to ``block``, as
    ``{"block", "index", "key", "value"}``. Raises on RPC failure."""
    from web3 import Web3

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(dao_registry_address),
        abi=_REGISTRY_EVENTS_ABI,
    )
    return [
        {
            "block": int(log["blockNumber"]),
            "index": int(log["logIndex"]),
            "key": log["args"]["key"],
            "value": log["args"]["value"],
        }
        for log in _get_logs(
            contract.events.RegistryUpdated, w3, contract.address, block,
            start_block)
    ]


def read_pinned_charter_text(
    w3: Any, dao_registry_address: str, block: int,
    key: str = CHARTER_REGISTRY_KEY,
) -> Optional[str]:
    """The jurisdiction's charter text as of ``block`` (the snapshot
    anchor's block), from the DAO Registry's RegistryUpdated logs. Raises
    on RPC failure, like the rest of the voice read."""
    records = _registry_records(w3, dao_registry_address, block)
    return charter_text_from_logs(records, block, key)


# ---------------------------------------------------------------------------
# Sanctions (docs/accountability.md). A DAO proposal writes
# ``sanction.<household 0x>`` = JSON ``{"until": <unix seconds>, "reason":
# <uri>, ...}`` into the Registry; an empty value lifts it. A sanction is in
# force while the snapshot anchor's timestamp is before ``until`` (no
# ``until`` = until lifted). The key names the household (the proven owner
# wallet); naming one of its agents sanctions the same household, so a fresh
# agent key does not escape it. Pinned to the snapshot anchor like the
# charter, so every daemon's close reads the same list.
# ---------------------------------------------------------------------------

SANCTION_KEY_PREFIX = "sanction."


def parse_sanction(value: str) -> Optional[Dict[str, Any]]:
    """The sanction a Registry value describes, or None when it is empty
    (lifted). A non-empty value that is not a JSON object still sanctions,
    with no end and the raw text as its note: the DAO voted for it. A
    non-integer ``until`` is ignored (no end)."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return {"note": text}
    out = dict(data)
    until = out.get("until")
    if isinstance(until, bool) or not isinstance(until, int):
        out.pop("until", None)
    return out


def sanction_in_force(entry: Dict[str, Any], at_timestamp: int) -> bool:
    until = entry.get("until")
    return until is None or int(at_timestamp) < int(until)


def sanctions_from_records(
    records: List[Dict[str, Any]], block: int, at_timestamp: int,
) -> Dict[str, Dict[str, Any]]:
    """Sanctions in force as of ``block`` and ``at_timestamp``: address
    (lowercased 0x) -> parsed entry, from the last write of each
    ``sanction.<address>`` key at or before ``block``."""
    last: Dict[str, str] = {}
    for rec in sorted(records, key=lambda r: (int(r["block"]), int(r["index"]))):
        if int(rec["block"]) > int(block):
            break
        key = str(rec.get("key") or "")
        if key.startswith(SANCTION_KEY_PREFIX):
            addr = key[len(SANCTION_KEY_PREFIX):].strip().lower()
            if addr:
                last[addr] = str(rec.get("value") or "")
    out: Dict[str, Dict[str, Any]] = {}
    for addr in sorted(last):
        entry = parse_sanction(last[addr])
        if entry is not None and sanction_in_force(entry, at_timestamp):
            out[addr] = entry
    return out


# ---------------------------------------------------------------------------
# Robot policy (docs/robots.md, 2026-10-25). One jurisdiction is one
# territory. Its DAO Registry holds, by proposal:
#   jurisdiction.<id>.robots           its robots: JSON list (or comma list)
#   jurisdiction.<id>.robot_standards  required safety-pack digests: JSON list
#   jurisdiction.<id>.parent           the jurisdiction it is nested in:
#                                      JSON {"registry": "0x...", "id": "<id>"}
# Standards cascade: a robot must run its own jurisdiction's packs plus every
# ancestor's. Parents must live on the same chain; each is read at the SAME
# snapshot block, so every daemon derives the same set. A level can add
# standards, never drop an inherited one.
# ---------------------------------------------------------------------------

ROBOT_POLICY_MAX_DEPTH = 8


def _last_writes(records: List[Dict[str, Any]], block: int) -> Dict[str, str]:
    last: Dict[str, str] = {}
    for rec in sorted(records, key=lambda r: (int(r["block"]), int(r["index"]))):
        if int(rec["block"]) > int(block):
            break
        last[str(rec.get("key") or "")] = str(rec.get("value") or "")
    return last


def _json_list(text: str, field: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        items = data.get(field, []) if isinstance(data, dict) else data
    except ValueError:
        items = text.split(",")
    return [str(x).strip().lower() for x in items] if isinstance(items, list) else []


def robot_policy(
    fetch: Any, registry: str, block: int, jurisdiction_id: str = "autonet",
) -> Dict[str, Any]:
    """``{"robots", "robot_standards", "jurisdiction_lineage"}`` as of
    ``block``. ``fetch(registry_address)`` returns that Registry's records
    (``_registry_records``). The lineage is the jurisdiction ids from this
    one up, as walked (stops at a loop or ROBOT_POLICY_MAX_DEPTH)."""
    robots: List[str] = []
    standards: set = set()
    lineage: List[str] = []
    seen: set = set()
    reg, jid = str(registry).lower(), jurisdiction_id
    while reg and reg not in seen and len(lineage) < ROBOT_POLICY_MAX_DEPTH:
        seen.add(reg)
        lineage.append(jid)
        last = _last_writes(fetch(reg), block)
        prefix = f"jurisdiction.{jid}."
        if len(lineage) == 1:
            robots = sorted({a for a in _json_list(last.get(prefix + "robots", ""), "robots")
                             if a.startswith("0x")})
        standards |= {d.removeprefix("0x") for d in
                      _json_list(last.get(prefix + "robot_standards", ""), "packs")
                      if len(d.removeprefix("0x")) == 64}
        try:
            parent = json.loads(last.get(prefix + "parent", "") or "null")
        except ValueError:
            parent = None
        if not isinstance(parent, dict):
            break
        reg = str(parent.get("registry") or "").lower()
        jid = str(parent.get("id") or "")
        if not reg.startswith("0x") or not jid:
            break
    return {"robots": robots, "robot_standards": sorted(standards),
            "jurisdiction_lineage": lineage}


# ---------------------------------------------------------------------------
# Epoch window (Decision 2026-10-26, items 1-4 and 7).
#
# Cutoffs run on their own schedule: epoch n+1 ends at epoch n's cutoff
# plus its length (Substrate.deployedAt stands in for the cutoff before
# epoch 1). Its batches are those signed after epoch n's cutoff and at or
# before its own. An empty window slides the cutoff forward one grace step
# at a time until a batch falls in it. Each epoch's cutoff is part of its
# id, so the next window's start is read from the chain (the last anchor's
# epochId), not from local state.
#
# An epoch ends on VOLUME (Decision 2026-10-27): at the first moment after
# the previous cutoff when the fees burned since it reach
# ``epoch_target_share`` of the ATN supply (pinned at the last anchor),
# read from Substrate's on-chain burn checkpoints (``burnedBetween``), so
# every daemon finds the same moment. Bounded below by one DAO proposal
# lifecycle (voting delay + voting period + timelock delay: an epoch cannot
# be anchored faster) and above by ``epoch_ceiling_votes`` voting periods,
# so a quiet network still pays out. Integer arithmetic. (This replaced a
# Bitcoin-style retarget from previous epochs' fee rate.)
# ---------------------------------------------------------------------------

EPOCH_RULE_DEFAULTS: Dict[str, int] = {
    # Share of the ATN supply one epoch should burn in fees, in parts per
    # million: the volume that ends an epoch. 0 = unset (set by the DAO at
    # deployment): every epoch lasts the floor.
    "epoch_target_share": 0,
    # Retired with the retarget (volume ends an epoch now); still read so
    # an old Registry key is harmless.
    "epoch_retarget_window": 4,
    # Ceiling on the length, in voting periods.
    "epoch_ceiling_votes": 10,
    # Seconds a daemon waits past the cutoff before closing, so batches
    # signed just before it can arrive.
    "epoch_grace": 120,
    # Largest gap allowed between a batch's signed timestamp and its
    # arrival; anything further off is refused (no backdating into a
    # closing epoch). Must stay below the grace.
    "epoch_skew": 60,
}

EPOCH_ID_PREFIX = "epoch"
SHARE_SCALE = 1_000_000

_GOVERNOR_EVENTS_ABI = list((
        {"anonymous": False, "type": "event", "name": "VotingDelaySet",
         "inputs": [
             {"indexed": False, "internalType": "uint256",
              "name": "oldVotingDelay", "type": "uint256"},
             {"indexed": False, "internalType": "uint256",
              "name": "newVotingDelay", "type": "uint256"}]},
        {"anonymous": False, "type": "event", "name": "VotingPeriodSet",
         "inputs": [
             {"indexed": False, "internalType": "uint256",
              "name": "oldVotingPeriod", "type": "uint256"},
             {"indexed": False, "internalType": "uint256",
              "name": "newVotingPeriod", "type": "uint256"}]},
        {"anonymous": False, "type": "event", "name": "TimelockChange",
         "inputs": [
             {"indexed": False, "internalType": "address",
              "name": "oldTimelock", "type": "address"},
             {"indexed": False, "internalType": "address",
              "name": "newTimelock", "type": "address"}]},
))
_TIMELOCK_EVENTS_ABI = [
    {"anonymous": False, "type": "event", "name": "MinDelayChange",
     "inputs": [
         {"indexed": False, "internalType": "uint256",
          "name": "oldDuration", "type": "uint256"},
         {"indexed": False, "internalType": "uint256",
          "name": "newDuration", "type": "uint256"}]},
]


def epoch_rule_params(
    records: List[Dict[str, Any]], block: Optional[int],
    jurisdiction_id: str = "autonet",
) -> Dict[str, int]:
    """The governed epoch rules in force at ``block`` (Registry keys
    ``jurisdiction.<id>.<name>`` for each ``EPOCH_RULE_DEFAULTS`` name,
    whole numbers). Unset, malformed or negative values fall back to the
    defaults; ``block`` None (no anchor yet) means the defaults."""
    writes = _last_writes(records, int(block)) if block is not None else {}
    out: Dict[str, int] = {}
    for name, default in EPOCH_RULE_DEFAULTS.items():
        raw = writes.get(f"jurisdiction.{jurisdiction_id}.{name}")
        try:
            value = int(str(raw).strip()) if raw is not None else default
        except ValueError:
            value = default
        out[name] = value if value >= 0 else default
    return out


def make_epoch_id(number: int, prev_root_hex: str, cutoff: int) -> str:
    """``epoch-<n>-<first 16 hex of anchor n-1's root>-<cutoff>``."""
    root = str(prev_root_hex or "").lower().removeprefix("0x")
    return f"{EPOCH_ID_PREFIX}-{int(number)}-{root[:16]:0<16}-{int(cutoff)}"


def epoch_id_cutoff(epoch_id: str) -> Optional[int]:
    """The cutoff an epoch id carries, or None for an id not in the
    ``make_epoch_id`` form."""
    parts = str(epoch_id).split("-")
    if len(parts) != 4 or parts[0] != EPOCH_ID_PREFIX:
        return None
    try:
        return int(parts[3])
    except ValueError:
        return None


def governor_lifecycle(
    w3: Any, governor_address: str, block: Optional[int],
    start_block: Optional[int] = None,
) -> Dict[str, int]:
    """{voting_delay, voting_period, min_delay} in seconds (the DAO clock
    is timestamp mode) as of ``block`` (None = latest), rebuilt from the
    governor's and timelock's settings events: they fire at construction
    and on every change, so no archive read is needed. Zeros when no
    governor is configured (a dev network anchoring from an EOA)."""
    out = {"voting_delay": 0, "voting_period": 0, "min_delay": 0}
    if not governor_address:
        return out
    from web3 import Web3

    to_block = int(block) if block is not None else int(w3.eth.block_number)
    gov = w3.eth.contract(address=Web3.to_checksum_address(governor_address),
                          abi=_GOVERNOR_EVENTS_ABI)

    def _last(event, address, field):
        logs = sorted(
            _get_logs(event, w3, address, to_block, start_block),
            key=lambda log: (int(log["blockNumber"]), int(log["logIndex"])))
        return logs[-1]["args"][field] if logs else None

    delay = _last(gov.events.VotingDelaySet, gov.address, "newVotingDelay")
    period = _last(gov.events.VotingPeriodSet, gov.address, "newVotingPeriod")
    timelock = _last(gov.events.TimelockChange, gov.address, "newTimelock")
    out["voting_delay"] = int(delay or 0)
    out["voting_period"] = int(period or 0)
    if timelock and str(timelock).lower() != _ZERO:
        tl = w3.eth.contract(address=Web3.to_checksum_address(timelock),
                             abi=_TIMELOCK_EVENTS_ABI)
        out["min_delay"] = int(
            _last(tl.events.MinDelayChange, tl.address, "newDuration") or 0)
    return out


def volume_cutoff(
    burned_between: Callable[[int, int], int], *, start: int, floor: int,
    ceiling: int, target_share_ppm: int, supply: int, final_ts: int,
) -> Optional[int]:
    """The cutoff of the epoch that began at ``start``, or None while it is
    still open.

    It is the first second t in [start + floor, start + ceiling] at which
    ``burned_between(start, t)`` (Substrate's on-chain burn checkpoints)
    reaches ``target_share_ppm`` of ``supply``. Only time at or before
    ``final_ts`` is read, so every daemon sees the same, final burns and
    lands on the same second. Unset target (or no supply) => start + floor.
    A quiet network still closes at start + ceiling."""
    floor = max(0, int(floor))
    ceiling = max(floor, int(ceiling))
    lo, hi_cap = int(start) + floor, int(start) + ceiling
    threshold = (int(target_share_ppm) * int(supply)) // SHARE_SCALE
    if target_share_ppm <= 0 or threshold <= 0:
        return lo if final_ts >= lo else None
    hi = min(int(final_ts), hi_cap)
    if hi < lo:
        return None
    if int(burned_between(int(start), hi)) < threshold:
        return hi_cap if final_ts >= hi_cap else None
    if int(burned_between(int(start), lo)) >= threshold:
        return lo
    # burned(lo) < threshold <= burned(hi): smallest t with burned(t) >= it.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if int(burned_between(int(start), mid)) >= threshold:
            hi = mid
        else:
            lo = mid
    return hi


def epoch_window(contract: Any, *, number: int, prev_root: str,
                 start: Optional[int], params_block: Optional[int],
                 records: List[Dict[str, Any]],
                 jurisdiction_id: str = "autonet",
                 lifecycle: Optional[Dict[str, int]] = None,
                 history: Optional[List[Tuple[int, int]]] = None,
                 supply: int = 0) -> Dict[str, Any]:
    """The window of epoch ``number``.

    ``prev_root`` is the hex root of epoch number-1 and ``start`` its
    cutoff (None for epoch 1, which counts from ``Substrate.deployedAt``);
    ``params_block`` the block the governed rules are read at (None = the
    defaults); ``records`` the DAO Registry writes; ``lifecycle`` the
    governor settings (``governor_lifecycle``); ``history`` the earlier
    epochs as (cutoff, fees burned in that epoch's window), oldest first,
    ending with epoch number-1; ``supply`` the ATN supply at the pin.
    Returns {number, prev_root, start, base, length, floor, ceiling, grace,
    skew, target_share, supply}: with a target share set the epoch ends on
    volume (``volume_cutoff``), ``length`` being its floor; ``history`` is
    no longer used."""
    deployed_at = int(contract.functions.deployedAt().call())
    lc = lifecycle or {}
    params = epoch_rule_params(records, params_block, jurisdiction_id)
    base = deployed_at if start is None else int(start)
    floor = (int(lc.get("voting_delay", 0)) + int(lc.get("voting_period", 0))
             + int(lc.get("min_delay", 0)))
    ceiling = params["epoch_ceiling_votes"] * int(lc.get("voting_period", 0))
    return {
        "number": int(number),
        "prev_root": str(prev_root),
        "start": None if start is None else int(start),
        "base": base,
        "length": floor,
        "floor": floor,
        "ceiling": max(floor, ceiling),
        "grace": params["epoch_grace"],
        "skew": params["epoch_skew"],
        "target_share": int(params["epoch_target_share"]),
        "supply": int(supply),
    }


def _anchor_cutoff(anchor: Any) -> int:
    """An anchor's cutoff: its own field, else the one its id carries,
    else (an anchor from before cutoffs) its timestamp."""
    if len(anchor) > 13 and int(anchor[13]) > 0:
        return int(anchor[13])
    c = epoch_id_cutoff(str(anchor[0]))
    return int(anchor[8]) if c is None else c


def _window_after(w3: Any, contract: Any, anchor_count: int, anchor: Any,
                  records: List[Dict[str, Any]], jurisdiction_id: str,
                  governor_address: str,
                  start_block: Optional[int]) -> Dict[str, Any]:
    """The window after the latest anchor, every input pinned to it."""
    block = int(anchor[7])
    return epoch_window(
        contract, number=anchor_count + 1, prev_root=bytes(anchor[1]).hex(),
        start=_anchor_cutoff(anchor), params_block=block, records=records,
        jurisdiction_id=jurisdiction_id,
        lifecycle=governor_lifecycle(w3, governor_address, block, start_block),
        supply=int(contract.functions.atnTotalSupplyAt(block).call()))


def _window_pinned(w3: Any, contract: Any, pin: Dict[str, Any], block: int,
                   records: List[Dict[str, Any]], jurisdiction_id: str,
                   governor_address: str,
                   start_block: Optional[int]) -> Dict[str, Any]:
    """The window of epoch ``pin["number"]``, every input pinned to the
    previous epoch's cutoff (``block`` = the last block at or before it)."""
    cutoffs = [int(c) for c in pin["cutoffs"]]
    return epoch_window(
        contract, number=int(pin["number"]), prev_root=str(pin["prev_root"]),
        start=cutoffs[-1], params_block=block, records=records,
        jurisdiction_id=jurisdiction_id,
        lifecycle=governor_lifecycle(w3, governor_address, block, start_block),
        supply=int(contract.functions.atnTotalSupplyAt(block).call()))


_BLOCK_AT_CACHE: Dict[Any, int] = {}


def block_at_or_before(w3: Any, timestamp: int, lo: int = 0) -> int:
    """The last block whose timestamp is at or before ``timestamp`` (binary
    search over headers, which any node serves). Raises when the chain head
    is not yet past ``timestamp``: the block could still change."""
    key = (id(w3.provider), int(timestamp))
    hit = _BLOCK_AT_CACHE.get(key)
    if hit is not None:
        return hit
    hi = int(w3.eth.block_number)
    if int(w3.eth.get_block(hi)["timestamp"]) <= int(timestamp):
        raise ValueError(f"chain head not past {timestamp} yet")
    lo = max(0, int(lo))
    if int(w3.eth.get_block(lo)["timestamp"]) > int(timestamp):
        raise ValueError(f"no block at or before {timestamp}")
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if int(w3.eth.get_block(mid)["timestamp"]) <= int(timestamp):
            lo = mid
        else:
            hi = mid
    _BLOCK_AT_CACHE[key] = lo
    return lo


def sanction_since_blocks(records: List[Dict[str, Any]],
                          block: int) -> Dict[str, int]:
    """The block each sanction in the records was last written at (at or
    before ``block``): address -> block."""
    out: Dict[str, int] = {}
    for rec in sorted(records, key=lambda r: (int(r["block"]), int(r["index"]))):
        if int(rec["block"]) > int(block):
            break
        key = str(rec.get("key") or "")
        if key.startswith(SANCTION_KEY_PREFIX):
            out[key[len(SANCTION_KEY_PREFIX):].strip().lower()] = int(rec["block"])
    return out


def sanctioned_households(
    sanctions: Iterable[str], owner_of: Dict[str, str],
    rotations: Optional[List[Tuple[int, str, str, str]]] = None,
    since_block: Optional[Dict[str, int]] = None,
) -> List[str]:
    """The households the sanctioned addresses name: an agent's owner when
    it is bound, the address itself otherwise. Sorted, lowercased.

    ``rotations`` (block, agent, previous owner, new owner) close the
    rotation escape
    (survey 2026-10-26): an agent rotated away from a sanctioned household
    at or after its sanction was written carries the sanction to the
    household it moved to. Rotation needs the new owner's signature, so
    that household accepted a sanctioned agent; it is sanctioned from the
    rotation on, and the rule repeats for agents it rotates on."""
    since: Dict[str, int] = {}
    out = set()
    for addr in sanctions:
        a = str(addr).lower()
        owner = str(owner_of.get(a, "") or "").lower()
        house = owner if owner and owner != _ZERO else a
        out.add(house)
        start = int((since_block or {}).get(a, 0))
        since[house] = min(since.get(house, start), start)
    changed = bool(rotations)
    while changed:
        changed = False
        for blk, agent, prev, new in rotations or ():
            prev = str(prev).lower()
            if prev not in out or int(blk) < since.get(prev, 0):
                continue
            cur = str(new).lower() or str(agent).lower()
            if cur not in out:
                out.add(cur)
                since[cur] = int(blk)
                changed = True
    return sorted(out)


def read_voice_state(
    substrate_address: str,
    rpc_url: str = "",
    *,
    epsilon: float = VOICE_EPSILON,
    rep_token_address: str = "",
    tool_bond_address: str = "",
    dao_registry_address: str = "",
    web3: Any = None,
    anchor_index: Optional[int] = None,
    jurisdiction_id: str = "autonet",
    deploy_block: Optional[int] = None,
    governor_address: str = "",
    pin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read {owner_map, voice_weights, rep_shares, rep_supply, supply,
    snapshot_block, emission_pool, recycled, weight_source,
    review_snapshot} as of the previous epoch's anchor.

    ``review_snapshot`` (docs/multi_jurisdiction.md decision 10): ToolBond
    ratings and the reviewer-share rounds, rebuilt from ToolBond logs up to
    the SAME snapshot block as the owner map (``read_tool_review_snapshot``),
    with the rounds finalized since the anchor before it. ``supply`` (REP)
    rides along for the close's mass normalization. Empty (``tools == {}``)
    when ``tool_bond_address`` is unset or at genesis.

    ``charter_text`` (docs/multi_jurisdiction.md decision 5): the DAO
    Registry's ``jurisdiction.autonet.charter`` value as of the SAME
    snapshot block (``read_pinned_charter_text``). None when
    ``dao_registry_address`` is unset, at genesis, or never written.

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
    it and blocks the close.

    ``deploy_block`` (the jurisdiction's configured deployment block)
    starts every log scan; None finds each contract's creation block.

    ``pin`` (Decision 2026-10-26, pipelined elections) reads the inputs of
    epoch ``pin["number"]`` as of the previous epoch's CUTOFF instead of
    the latest anchor, so an epoch can close before the anchor before it
    lands. ``pin["cutoffs"]`` are the cutoffs of epochs 1..number-1 (the
    agreed ones plus this daemon's own view of any unanchored ones) and
    ``pin["prev_root"]`` the root of epoch number-1. REP is read at that
    cutoff, logs up to the last block at or before it, and the pool is the
    fees burned between the two cutoffs before it (lag-1, the window the
    chain records as that epoch's windowBurned).
    """
    start = int(deploy_block) if deploy_block else None
    from web3 import Web3

    w3 = web3 if web3 is not None else Web3(Web3.HTTPProvider(rpc_url))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(substrate_address), abi=_VOICE_ABI,
    )

    anchor_count = contract.functions.anchorCount().call()
    pinned = pin is not None and int(pin["number"]) > 1
    if pin is not None and not pinned:
        anchor_count = 0
    if anchor_index is not None:
        # Read as if the chain ended at that anchor (the robot poll rail
        # pins a round to the anchor in force when the round started).
        anchor_count = max(0, min(int(anchor_count), int(anchor_index) + 1))
    if anchor_count == 0 and not pinned:
        # No agreed snapshot yet (epoch 1): no voices, empty fees-only pool.
        # (A pinned read of a later epoch never takes this branch: it may
        # run before any anchor has landed.)
        return {
            "epoch_window": epoch_window(
                contract, number=1, prev_root="00" * 32, start=None,
                params_block=None, records=[],
                jurisdiction_id=jurisdiction_id,
                lifecycle=governor_lifecycle(
                    w3, governor_address, None, start)),
            "owner_map": {},
            "voice_weights": {},
            "rep_shares": {},
            "rep_supply": 0,
            "supply": 0,
            "snapshot_block": None,
            "emission_pool": 0.0,
            "recycled": 0.0,
            "emission_pool_raw": 0,
            "snapshot_anchor_count": 0,
            "weight_source": "reputation",
            "review_snapshot": {
                "reviewer_share_bps": REVIEWER_SHARE_BPS_DEFAULT,
                "tools": {}, "rounds": {}, "supply": 0},
            "charter_text": None,
            "sanctions": {},
            "sanctioned_households": [],
            "agent_rep": {},
            "robots": [],
            "robot_standards": [],
            "jurisdiction_lineage": [],
        }
    if pinned:
        cutoffs = [int(c) for c in pin["cutoffs"]]
        number = int(pin["number"])
        if not cutoffs or len(cutoffs) > number - 1:
            raise ValueError(f"pin for epoch {number} needs 1..{number - 1} "
                             f"cutoffs (the latest ones), got {len(cutoffs)}")
        anchor = None
        anchor_count = number - 1
        timestamp = cutoffs[-1]
        block = block_at_or_before(w3, timestamp, start or 0)
        deployed_at = int(contract.functions.deployedAt().call())
        prev = (cutoffs[-2] if len(cutoffs) > 1 else
                deployed_at if number == 2 else None)
        if prev is None:
            raise ValueError("pin needs the two latest cutoffs past epoch 2")
        # The pool this epoch may mint: what the chain records as epoch
        # number-1's windowBurned (fees between its two cutoffs).
        burned_raw = int(contract.functions.burnedBetween(prev, timestamp).call())
    else:
        anchor = contract.functions.getAnchor(anchor_count - 1).call()
        block = int(anchor[7])       # Anchor.blockNumber (struct field 7)
        timestamp = int(anchor[8])   # Anchor.timestamp (struct field 8)
        # Fees-only emission, read from the SAME on-chain quantity
        # Substrate caps mints with (2026-10-12): the snapshot anchor's
        # windowBurned. Substrate sets the next anchor's mintPool to
        # exactly this value, so the close's pool and the on-chain cap
        # cannot disagree.
        burned_raw = int(anchor[10]) if len(anchor) > 10 else 0
    recycled = burned_raw / _ATN_SCALE

    # Agent set + owner map from event logs up to the snapshot block —
    # deterministic (chain history) and served by non-archive nodes.
    # Last OwnerBound per agent wins (rotation support), ordered by
    # (blockNumber, logIndex).
    agents = sorted({
        str(log["args"]["agent"]).lower()
        for log in _get_logs(
            contract.events.AgentRegistered, w3, contract.address, block,
            start)
    })
    bindings = sorted(
        _get_logs(contract.events.OwnerBound, w3, contract.address, block,
                  start),
        key=lambda log: (int(log["blockNumber"]), int(log["logIndex"])),
    )
    owner_of: Dict[str, str] = {}
    parent_of: Dict[str, str] = {}
    for log in bindings:
        agent_l = str(log["args"]["agent"]).lower()
        owner_of[agent_l] = str(log["args"]["owner"]).lower()
        parent_of[agent_l] = str(log["args"]["parent"]).lower()

    # Revoked agents drop out with their whole subtree (Substrate.isLive
    # walks the same parent links), as of the snapshot block.
    revoked = {
        str(log["args"]["agent"]).lower()
        for log in _get_logs(
            contract.events.AgentRevoked, w3, contract.address, block, start)
    }
    if revoked:
        def _live(agent: str) -> bool:
            seen = set()
            a = agent
            while a and a != _ZERO and a not in seen:
                if a in revoked:
                    return False
                seen.add(a)
                a = parent_of.get(a, "")
            return True
        agents = [a for a in agents if _live(a)]

    # Reputation (voice) reads: RepToken (DAO), pinned to the anchor
    # TIMESTAMP (ERC20Votes mode=timestamp). No RepToken configured =>
    # genesis regime: empty rep, close runs drift at weight 1.0.
    owner_map: Dict[str, str] = {}
    for agent in agents:
        owner = owner_of.get(agent, "")
        if owner and owner != _ZERO:
            owner_map[agent] = owner

    supply = 0
    agent_rep: Dict[str, int] = {}
    voice_weights: Dict[str, float] = {}
    rep_shares: Dict[str, float] = {}
    if rep_token_address:
        rep = w3.eth.contract(
            address=Web3.to_checksum_address(rep_token_address), abi=_REP_ABI,
        )
        # Denominator is REPUTATION total supply (voice), not ATN.
        supply = int(rep.functions.getPastTotalSupply(timestamp).call())

        def _reputation_at(addr: str) -> int:
            # The agent's own REP balance at the pinned timestamp, whatever
            # it delegated (see module docstring).
            return int(rep.functions.getPastBalance(
                Web3.to_checksum_address(addr), timestamp).call())

        # household key -> raw reputation sum (int, exact — floats only at
        # the final ratio so accumulation order can't jitter the weights).
        house_rep: Dict[str, int] = {}
        for agent in agents:
            owner = owner_of.get(agent, "")
            r = _reputation_at(agent)
            agent_rep[agent] = r
            if owner and owner != _ZERO:
                # Bound agent: REP rolls up to the owner household.
                house_rep[owner] = house_rep.get(owner, 0) + r
            else:
                house_rep[agent] = house_rep.get(agent, 0) + r
        # The owner wallet's own REP (it claims for the household). An
        # owner that is itself a registered agent was counted above.
        agent_set = set(agents)
        for owner in sorted(set(owner_map.values())):
            if owner not in agent_set:
                house_rep[owner] = house_rep.get(owner, 0) + _reputation_at(owner)

        # The RAW rep share (no epsilon floor) drives drift weight;
        # voice_weights keeps the epsilon floor for mint bootstrap.
        for house in sorted(house_rep.keys()):
            share = (house_rep[house] / supply) if supply > 0 else 0.0
            voice_weights[house] = round(epsilon + share, 9)
            rep_shares[house] = round(share, 9)

    # The DAO Registry pinned to the same anchor block: the charter in force
    # and the sanctions in force at the anchor's timestamp.
    charter_text: Optional[str] = None
    sanctions: Dict[str, Dict[str, Any]] = {}
    policy: Dict[str, Any] = {"robots": [], "robot_standards": [],
                              "jurisdiction_lineage": []}
    reg_records: List[Dict[str, Any]] = []
    if dao_registry_address:
        reg_records = _registry_records(
            w3, dao_registry_address, block, start)
        charter_text = charter_text_from_logs(reg_records, block)
        sanctions = sanctions_from_records(reg_records, block, timestamp)
        own = str(dao_registry_address).lower()
        policy = robot_policy(
            lambda reg: reg_records if reg == own
            else _registry_records(w3, reg, block, start),
            own, block, jurisdiction_id)
    rotations = [
        (int(log["blockNumber"]), str(log["args"]["agent"]).lower(),
         str(log["args"]["prevOwner"]).lower(),
         str(log["args"]["owner"]).lower())
        for log in bindings
        if str(log["args"].get("prevOwner") or _ZERO).lower() != _ZERO]
    houses = sanctioned_households(
        sanctions, owner_of, rotations,
        sanction_since_blocks(reg_records, block) if dao_registry_address else {})
    house_set = set(houses)
    # Every address whose household is sanctioned: its reviews move no
    # rating.
    excluded_reviewers = sorted(
        {a for a in agents if (owner_of.get(a) or a) in house_set}
        | house_set | set(sanctions))

    # ToolBond review snapshot, pinned to the same anchor block; the
    # reviewer-share window opens at the anchor before it (block 0 = since
    # deployment when this is the first anchor).
    review_snapshot: Dict[str, Any] = {
        "reviewer_share_bps": REVIEWER_SHARE_BPS_DEFAULT,
        "tools": {}, "rounds": {}}
    if tool_bond_address:
        since_block = 0
        if pinned and len(pin["cutoffs"]) >= 2:
            since_block = block_at_or_before(
                w3, int(pin["cutoffs"][-2]), start or 0)
        elif not pinned and anchor_count >= 2:
            since_block = int(contract.functions.getAnchor(anchor_count - 2).call()[7])
        review_snapshot = read_tool_review_snapshot(
            w3, tool_bond_address, block, since_block=since_block,
            excluded_reviewers=excluded_reviewers, start_block=start)
    review_snapshot["supply"] = int(supply)

    return {
        "epoch_window": (
            _window_pinned(w3, contract, pin, block, reg_records,
                           jurisdiction_id, governor_address, start)
            if pinned else _window_after(
                w3, contract, anchor_count, anchor, reg_records,
                jurisdiction_id, governor_address, start)),
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
        # Exact on-chain pool (base units) + which anchor it was read at:
        # the anchor this close produces is valid only as anchor number
        # snapshot_anchor_count (else its on-chain mintPool differs).
        "emission_pool_raw": burned_raw,
        "snapshot_anchor_count": int(anchor_count),
        "weight_source": "reputation",
        "review_snapshot": review_snapshot,
        "charter_text": charter_text,
        # Sanctions in force at the snapshot (docs/accountability.md): the
        # raw entries, and the households they name. The close withholds
        # those households' mint and their drift weight.
        "sanctions": sanctions,
        "sanctioned_households": houses,
        # Containment thresholds in force at the snapshot: the close
        # mints nothing for a tool they deny (Decision 2026-10-27).
        "containment_params": params_from_writes(
            _last_writes(reg_records, block) if reg_records else {}, jurisdiction_id),
        # Per-agent REP (raw base units) and the robot policy (docs/robots.md,
        # 2026-10-25): the robot poll rail draws pollers from the REP, takes
        # witnesses from the fleet and checks packs against the cascaded
        # standards, all at the same snapshot.
        "agent_rep": agent_rep,
        **policy,
    }
