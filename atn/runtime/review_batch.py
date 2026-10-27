"""Staked reviews on ToolBond (docs/multi_jurisdiction.md decision 10).

Every review of a tool is one entry in its agent's batch for the epoch. The
agent commits the batch once per epoch (a hash plus the total ATN stake),
and reveals it after the epoch is anchored, within ToolBond's reveal window.
ToolBond settles each tool's round on chain: the REP-weighted mean per axis
and harm, each reviewer's accuracy, stake gains and losses, REP penalties
and the author's bond slash (trustless-contracts ToolBond, 448023a).

A reveal is relayable: anyone holding the plaintext and the salt can submit
it, so the batch is kept on disk as a sealed reveal the owner can hand to a
backup. A batch never revealed forfeits its whole stake and the full REP
penalty, so revealing always beats withholding.

This module is the daemon's half: encoding, the commitment (computed locally,
identical to ToolBond.commitmentOf), a per-agent per-epoch batch store, and
the chain calls. The review step queues entries (``queue_reviews``, called
from attest_tools and vet_tool in atn/agent_tools.py), atn/review_driver.py
commits, reveals, finalizes and claims, and the close reads the settled
ratings from ToolBond's logs (nodes/common/voice_state.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

NO_SCORE = -128          # int8 sentinel: no score on this axis
NO_HARM = 255            # uint8 sentinel: no harm score
MAX_AXES = 16

REVIEW_INPUT_TYPE = "(bytes32,bytes,uint8,uint256,bytes32)"


def _fn(name: str, inputs: list, outputs: list = (), mutability: str = "nonpayable") -> dict[str, Any]:
    def arg(n: str, t: str) -> dict[str, Any]:
        if t == "ReviewInput[]":
            return {"name": n, "type": "tuple[]", "components": [
                {"name": "digest", "type": "bytes32"}, {"name": "scores", "type": "bytes"},
                {"name": "harm", "type": "uint8"}, {"name": "stake", "type": "uint256"},
                {"name": "evidenceHash", "type": "bytes32"}]}
        return {"name": n, "type": t}
    return {"name": name, "type": "function", "stateMutability": mutability,
            "inputs": [arg(n, t) for n, t in inputs],
            "outputs": [{"name": n, "type": t} for n, t in outputs]}


REVIEW_TUPLE = [
    ("reviewer", "address"), ("harm", "uint8"), ("claimed", "bool"),
    ("accuracy", "uint64"), ("scores", "bytes16"), ("stake", "uint128"),
    ("weight", "uint128"), ("harmAccuracy", "uint64"),
]


def _review_fn(name: str, inputs: list) -> dict[str, Any]:
    return {"name": name, "type": "function", "stateMutability": "view",
            "inputs": [{"name": n, "type": t} for n, t in inputs],
            "outputs": [{"name": "", "type": "tuple", "components": [
                {"name": n, "type": t} for n, t in REVIEW_TUPLE]}]}


# ToolBond.Stage
STAGE_NONE, STAGE_OPEN, STAGE_COOLING, STAGE_FINALIZING, STAGE_DONE = range(5)
STAGES = ("none", "open", "cooling", "finalizing", "done")
# Review.accuracy when nothing could be scored (refunded, no gain or loss).
UNINFORMATIVE = 2 ** 64 - 1

TOOL_BOND_REVIEW_ABI = [
    _fn("currentEpoch", [], [("", "uint256")], "view"),
    _fn("epochInfo", [("epoch", "uint256")],
        [("started", "bool"), ("charterHash", "bytes32"), ("axisIds", "bytes32[]"),
         ("snapshot", "uint48"), ("supply", "uint256"), ("revealDeadline", "uint256")], "view"),
    _fn("getRound", [("digest", "bytes32"), ("epoch", "uint256")],
        [("stage", "uint8"), ("overruled", "bool"), ("appealEnds", "uint48"),
         ("weight", "uint256"), ("harmSum", "uint256"), ("harmMass", "uint256"),
         ("settledHarm", "uint256"), ("stakes", "uint256"), ("reviewCount", "uint256")], "view"),
    _fn("roundAxis", [("digest", "bytes32"), ("epoch", "uint256"), ("i", "uint256")],
        [("sum", "int256"), ("mass", "uint256")], "view"),
    _review_fn("reviewAt", [("digest", "bytes32"), ("epoch", "uint256"), ("i", "uint256")]),
    _review_fn("reviewOf", [("digest", "bytes32"), ("epoch", "uint256"), ("reviewer", "address")]),
    _fn("isBonded", [("digest", "bytes32")], [("", "bool")], "view"),
    _fn("reviewerShareBps", [], [("", "uint256")], "view"),
    _fn("revealEnds", [("epoch", "uint256")], [("", "uint256")], "view"),
    _fn("commits", [("", "uint256"), ("", "address")],
        [("commitment", "bytes32"), ("stake", "uint128"), ("weight", "uint128"),
         ("revealed", "bool"), ("forfeited", "bool")], "view"),
    _fn("commitmentOf", [("epoch", "uint256"), ("agent", "address"),
                         ("reviews", "ReviewInput[]"), ("salt", "bytes32")],
        [("", "bytes32")], "pure"),
    # trustless-contracts ad32e38: the commit declares its review count
    # (public), the reveal must hold exactly that many.
    _fn("commit", [("commitment", "bytes32"), ("stake", "uint256"), ("reviews", "uint256")]),
    _fn("reveal", [("epoch", "uint256"), ("agent", "address"),
                   ("reviews", "ReviewInput[]"), ("salt", "bytes32")]),
    _fn("forfeit", [("epoch", "uint256"), ("agent", "address")]),
    _fn("finalize", [("digest", "bytes32"), ("epoch", "uint256"), ("maxReviews", "uint256")]),
    _fn("claim", [("digest", "bytes32"), ("epoch", "uint256"), ("reviewer", "address")]),
    _fn("toolRating", [("digest", "bytes32"), ("axisId", "bytes32")],
        [("sum", "int256"), ("mass", "uint256")], "view"),
    _fn("toolHarm", [("digest", "bytes32")], [("sum", "uint256"), ("mass", "uint256")], "view"),
]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_scores(axes: dict[str, float] | None, axis_ids: list[str]) -> bytes:
    """One int8 per charter axis, in charter order: round(score x 100),
    clamped to [-100, 100]; NO_SCORE where the review has no score. Empty
    bytes when the review scores no axis at all."""
    if not axes:
        return b""
    if len(axis_ids) > MAX_AXES:
        raise ValueError(f"a charter has at most {MAX_AXES} axes")
    out = bytearray()
    for axis_id in axis_ids:
        v = axes.get(axis_id)
        if v is None:
            out.append(NO_SCORE & 0xFF)
            continue
        s = max(-100, min(100, int(round(float(v) * 100))))
        out.append(s & 0xFF)
    if all(b == (NO_SCORE & 0xFF) for b in out):
        return b""
    return bytes(out)


def encode_harm(harm: float | None) -> int:
    """Harm in [0, 1] as 0..100; NO_HARM when absent."""
    if harm is None:
        return NO_HARM
    return max(0, min(100, int(round(float(harm) * 100))))


def _b32(h: str) -> bytes:
    s = str(h or "").lower().removeprefix("0x")
    if len(s) != 64:
        raise ValueError(f"expected a 32-byte hex value, got {h!r}")
    return bytes.fromhex(s)


@dataclass
class ReviewEntry:
    """One review in an agent's epoch batch."""
    digest: str                      # tool manifest digest (hex)
    axes: dict[str, float] = field(default_factory=dict)   # axis id -> [-1, 1]
    harm: Optional[float] = None     # [0, 1], None = no harm score
    stake: int = 0                   # ATN base units
    evidence_hash: str = "00" * 32   # hex

    def to_input(self, axis_ids: list[str]) -> tuple:
        return (_b32(self.digest), encode_scores(self.axes, axis_ids),
                encode_harm(self.harm), int(self.stake), _b32(self.evidence_hash))


def commitment_of(epoch: int, agent: str, inputs: list[tuple], salt: bytes) -> bytes:
    """keccak256(abi.encode(epoch, agent, reviews, salt)), as
    ToolBond.commitmentOf computes it."""
    from eth_abi import encode
    from eth_utils import keccak, to_checksum_address
    return keccak(encode(["uint256", "address", REVIEW_INPUT_TYPE + "[]", "bytes32"],
                         [int(epoch), to_checksum_address(agent), list(inputs), salt]))


# ---------------------------------------------------------------------------
# Batch store: one file per agent per epoch, under the jurisdiction's key
# ---------------------------------------------------------------------------

@dataclass
class EpochBatch:
    epoch: int
    agent: str                       # agent 0x
    axis_ids: list[str]
    reviews: list[ReviewEntry] = field(default_factory=list)
    salt: str = ""                   # hex, set at commit
    commitment: str = ""             # hex, set at commit
    committed_tx: str = ""
    revealed_tx: str = ""
    # Driver bookkeeping (atn/review_driver.py): "" pending, "committing"
    # (salt on disk, commit tx not confirmed), "committed", "revealed",
    # "settled" (every round claimed or nothing left to do), "forfeited"
    # (the reveal window closed unrevealed), "rolled" (reviews moved to a
    # later epoch because this one was anchored before a commit).
    status: str = ""
    # digest -> the round state this agent last saw: "open", "cooling",
    # "finalizing", "done", "claimed", or "none" (the review did not count).
    rounds: dict[str, str] = field(default_factory=dict)
    last_error: str = ""

    @property
    def total_stake(self) -> int:
        return sum(int(r.stake) for r in self.reviews)

    def inputs(self) -> list[tuple]:
        return [r.to_input(self.axis_ids) for r in self.reviews]

    def sealed_reveal(self) -> dict[str, Any]:
        """Everything a relayer needs to submit this batch's reveal."""
        return {"epoch": self.epoch, "agent": self.agent, "salt": self.salt,
                "axis_ids": list(self.axis_ids),
                "reviews": [asdict(r) for r in self.reviews]}


class ReviewBatchStore:
    """Pending and committed review batches on disk. A daemon runs one
    jurisdiction at a time, so batches live under its key."""

    def __init__(self, root: Path, jurisdiction: str) -> None:
        safe = "".join(c if c.isalnum() else "_" for c in (jurisdiction or "local"))
        self._dir = Path(root) / "review_batches" / safe

    def _path(self, agent: str, epoch: int) -> Path:
        return self._dir / f"{agent.lower()}_{int(epoch)}.json"

    def load(self, agent: str, epoch: int, axis_ids: list[str]) -> EpochBatch:
        p = self._path(agent, epoch)
        if not p.is_file():
            return EpochBatch(epoch=int(epoch), agent=agent, axis_ids=list(axis_ids))
        d = json.loads(p.read_text(encoding="utf-8"))
        return EpochBatch(epoch=int(d["epoch"]), agent=d["agent"], axis_ids=list(d["axis_ids"]),
                          reviews=[ReviewEntry(**r) for r in d.get("reviews", [])],
                          salt=d.get("salt", ""), commitment=d.get("commitment", ""),
                          committed_tx=d.get("committed_tx", ""),
                          revealed_tx=d.get("revealed_tx", ""),
                          status=d.get("status", ""),
                          rounds=dict(d.get("rounds") or {}),
                          last_error=d.get("last_error", ""))

    def save(self, batch: EpochBatch) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        p = self._path(batch.agent, batch.epoch)
        tmp = p.with_suffix(".tmp")
        d = {**asdict(batch), "reviews": [asdict(r) for r in batch.reviews]}
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        os.replace(tmp, p)

    def add(self, agent: str, epoch: int, axis_ids: list[str], entry: ReviewEntry) -> EpochBatch:
        """Queue a review for this epoch. A batch already committed is
        sealed; the review then waits for the next epoch's batch. A second
        review of the same tool in one epoch replaces the first."""
        batch = self.load(agent, epoch, axis_ids)
        if batch.commitment:
            batch = self.load(agent, epoch + 1, axis_ids)
        batch.reviews = [r for r in batch.reviews if r.digest != entry.digest] + [entry]
        self.save(batch)
        return batch

    def all(self) -> list[EpochBatch]:
        """Every stored batch, oldest epoch first."""
        out = []
        if not self._dir.is_dir():
            return out
        for p in sorted(self._dir.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append(self.load(d["agent"], d["epoch"], d["axis_ids"]))
            except (OSError, ValueError, KeyError) as exc:
                log.warning("unreadable review batch %s: %s", p.name, exc)
        return sorted(out, key=lambda b: (b.epoch, b.agent.lower()))

    def unrevealed(self) -> list[EpochBatch]:
        """Committed batches not yet revealed (oldest first)."""
        return [b for b in self.all()
                if b.commitment and not b.revealed_tx and b.status != "forfeited"]


# ---------------------------------------------------------------------------
# Chain calls
# ---------------------------------------------------------------------------

class ReviewBondClient:
    """Commit, reveal, forfeit, finalize and claim against ToolBond. Reuses
    ToolBondClient's address resolution and transaction plumbing."""

    def __init__(self, runtime: Any) -> None:
        from .tool_bond import ToolBondClient
        self._base = ToolBondClient(runtime)

    def available(self) -> bool:
        return self._base.available()

    def _contracts(self, w3):
        from web3 import Web3
        bond_addr, sub, _ = self._base._addresses()
        from .tool_bond import SUBSTRATE_TOOL_ABI
        return (w3.eth.contract(address=Web3.to_checksum_address(bond_addr), abi=TOOL_BOND_REVIEW_ABI),
                w3.eth.contract(address=Web3.to_checksum_address(sub), abi=SUBSTRATE_TOOL_ABI))

    async def _read(self, fn) -> Any:
        def run():
            return fn(*self._contracts(self._base._w3()))
        try:
            return await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            log.debug("ToolBond review read failed: %s", exc)
            return None

    async def _write(self, private_key: str, build, gas_floor: int = 300_000) -> dict[str, Any]:
        def run():
            from eth_account import Account
            w3 = self._base._w3()
            bond, substrate = self._contracts(w3)
            account = Account.from_key(private_key)
            return self._base._send(w3, account, build(bond, substrate, account), gas_floor)
        try:
            return await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            log.warning("ToolBond review write failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def current_epoch(self) -> Optional[int]:
        return await self._read(lambda b, s: int(b.functions.currentEpoch().call()))

    async def reveal_ends(self, epoch: int) -> Optional[int]:
        return await self._read(lambda b, s: int(b.functions.revealEnds(int(epoch)).call()))

    async def chain_time(self) -> Optional[int]:
        """The latest block's timestamp: the clock ToolBond's windows use."""
        def run():
            return int(self._base._w3().eth.get_block("latest")["timestamp"])
        try:
            return await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            log.debug("chain time read failed: %s", exc)
            return None

    async def commit_of(self, epoch: int, agent: str) -> Optional[dict[str, Any]]:
        def fn(b, s):
            from web3 import Web3
            c, stake, weight, revealed, forfeited = b.functions.commits(
                int(epoch), Web3.to_checksum_address(agent)).call()
            return {"commitment": c.hex(), "stake": int(stake), "weight": int(weight),
                    "revealed": bool(revealed), "forfeited": bool(forfeited)}
        return await self._read(fn)

    async def round_of(self, digest: str, epoch: int) -> Optional[dict[str, Any]]:
        def fn(b, s):
            r = b.functions.getRound(_b32(digest), int(epoch)).call()
            return {"stage": int(r[0]), "overruled": bool(r[1]), "appeal_ends": int(r[2]),
                    "weight": int(r[3]), "harm_sum": int(r[4]), "harm_mass": int(r[5]),
                    "settled_harm": int(r[6]), "stakes": int(r[7]), "reviews": int(r[8])}
        return await self._read(fn)

    async def review_of(self, digest: str, epoch: int,
                        reviewer: str) -> Optional[dict[str, Any]]:
        def fn(b, s):
            from web3 import Web3
            v = b.functions.reviewOf(_b32(digest), int(epoch),
                                     Web3.to_checksum_address(reviewer)).call()
            return {"reviewer": v[0], "harm": int(v[1]), "claimed": bool(v[2]),
                    "accuracy": int(v[3]), "stake": int(v[5]), "weight": int(v[6]),
                    "harm_accuracy": int(v[7])}
        return await self._read(fn)

    async def commit(self, private_key: str, batch: EpochBatch) -> dict[str, Any]:
        """Seal and commit ``batch`` (approving its stake first). The salt is
        fresh per batch; the caller saves the batch so the reveal survives a
        restart."""
        from eth_account import Account
        agent = Account.from_key(private_key).address
        if agent.lower() != batch.agent.lower():
            return {"success": False, "error": "batch belongs to another agent"}
        if not batch.salt:
            batch.salt = secrets.token_hex(32)
        c = commitment_of(batch.epoch, agent, batch.inputs(), bytes.fromhex(batch.salt))
        batch.commitment = c.hex()
        stake = batch.total_stake
        if stake > 0:
            allowance = await self._read(
                lambda b, s: int(s.functions.allowance(agent, b.address).call()))
            if allowance is None or allowance < stake:
                res = await self._write(private_key, lambda b, s, a: s.functions.approve(
                    b.address, int(stake)))
                if not res.get("success"):
                    return {**res, "step": "approve"}
        count = len(batch.inputs())
        res = await self._write(private_key, lambda b, s, a: b.functions.commit(
                                    c, int(stake), int(count)),
                                gas_floor=400_000)
        if res.get("success"):
            batch.committed_tx = res["tx_hash"]
        return res

    async def reveal(self, private_key: str, batch: EpochBatch) -> dict[str, Any]:
        """Submit ``batch``'s reveal. Any key can relay it; the agent named in
        the batch is the reviewer."""
        res = await self._write(private_key, lambda b, s, a: b.functions.reveal(
            int(batch.epoch), batch.agent, batch.inputs(), bytes.fromhex(batch.salt)),
            gas_floor=400_000)
        if res.get("success"):
            batch.revealed_tx = res["tx_hash"]
        return res

    async def forfeit(self, private_key: str, epoch: int, agent: str) -> dict[str, Any]:
        return await self._write(private_key, lambda b, s, a: b.functions.forfeit(int(epoch), agent))

    async def finalize(self, private_key: str, digest: str, epoch: int,
                       max_reviews: int = 50) -> dict[str, Any]:
        return await self._write(private_key, lambda b, s, a: b.functions.finalize(
            _b32(digest), int(epoch), int(max_reviews)), gas_floor=600_000)

    async def claim(self, private_key: str, digest: str, epoch: int,
                    reviewer: str) -> dict[str, Any]:
        return await self._write(private_key, lambda b, s, a: b.functions.claim(
            _b32(digest), int(epoch), reviewer))


# ---------------------------------------------------------------------------
# Queueing from the review step
# ---------------------------------------------------------------------------

def store_for(runtime: Any) -> ReviewBatchStore:
    """The runtime's batch store: root = its data dir, keyed by its
    jurisdiction (tests pre-set ``review_batch_store``)."""
    store = getattr(runtime, "review_batch_store", None)
    if isinstance(store, ReviewBatchStore):
        return store
    jurisdiction = ""
    try:
        jurisdiction = str(getattr(runtime, "jurisdiction", "") or "")
    except Exception:  # noqa: BLE001
        jurisdiction = ""
    store = ReviewBatchStore(Path(runtime._config.data_dir), jurisdiction)
    try:
        setattr(runtime, "review_batch_store", store)
    except Exception:  # noqa: BLE001
        pass
    return store


def client_for(runtime: Any) -> Any:
    """The runtime's review client (tests pre-set ``review_bond_client``)."""
    client = getattr(runtime, "review_bond_client", None)
    return client if client is not None else ReviewBondClient(runtime)


def normalize_review(raw: dict[str, Any], axis_ids: list[str]) -> Optional[ReviewEntry]:
    """A review step's {digest, axes?, harm?, stake?, evidence_hash?} as a
    ReviewEntry, or None when it carries nothing ToolBond can settle (no
    charter axis scored and no harm). Axes outside the charter are dropped;
    scores clamp to [-1, 1], harm to [0, 1], stake to a non-negative int."""
    axes: dict[str, float] = {}
    for axis_id, value in (raw.get("axes") or {}).items():
        if str(axis_id) not in axis_ids or isinstance(value, bool):
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v == v:
            axes[str(axis_id)] = max(-1.0, min(1.0, v))
    harm: Optional[float] = None
    raw_harm = raw.get("harm")
    if raw_harm is not None and not isinstance(raw_harm, bool):
        try:
            h = float(raw_harm)
            if h == h:
                harm = max(0.0, min(1.0, h))
        except (TypeError, ValueError):
            harm = None
    if not axes and harm is None:
        return None
    stake = 0
    try:
        stake = max(0, int(raw.get("stake") or 0))
    except (TypeError, ValueError):
        stake = 0
    evidence = str(raw.get("evidence_hash") or "").lower().removeprefix("0x")
    if len(evidence) != 64 or any(c not in "0123456789abcdef" for c in evidence):
        evidence = "00" * 32
    return ReviewEntry(digest=str(raw["digest"]).lower().removeprefix("0x"),
                       axes=dict(sorted(axes.items())), harm=harm, stake=stake,
                       evidence_hash=evidence)


async def queue_reviews(runtime: Any, agent_id: str,
                        reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Queue the reviewing agent's reviews into its batch for ToolBond's
    current epoch. Never raises; returns {queued, epoch, skipped, note}.
    Each review is {digest, axes?, harm?, stake?, evidence_hash?, author?};
    ``author`` (the manifest author) lets the agent's own tools be skipped.

    The agent must be registered on chain with a daemon-held key (the batch
    is committed with its key), and ToolBond must be reachable to read the
    epoch. Reviews carrying no charter axis score and no harm are skipped:
    ToolBond would refund them as uninformative."""
    from nodes.common.world_model_substrate.charter import active_charter

    out: dict[str, Any] = {"queued": 0, "epoch": None, "skipped": []}
    if not reviews:
        return out
    defn = runtime.get_agent(agent_id) if agent_id else None
    identity = getattr(defn, "identity", None) if defn else None
    address = str(getattr(identity, "address", "") or "")
    if not address or not getattr(identity, "registered_on_chain", False):
        out["note"] = ("reviews go on chain from a registered agent; this "
                       "caller is not registered, so nothing was queued")
        return out
    client = client_for(runtime)
    if not client.available():
        out["note"] = "no ToolBond configured; reviews were not queued"
        return out
    epoch = await client.current_epoch()
    if epoch is None:
        out["note"] = "ToolBond unreachable; reviews were not queued"
        return out
    axis_ids = list(active_charter().ids)
    store = store_for(runtime)
    batch = None
    for raw in reviews:
        if str(raw.get("author") or "").lower() == address.lower():
            # ToolBond never counts an author's review of their own tool;
            # queueing it would only lock the stake until the reveal.
            out["skipped"].append({"digest": raw.get("digest"),
                                   "error": "you cannot review your own tool"})
            continue
        try:
            entry = normalize_review(raw, axis_ids)
        except Exception as exc:  # noqa: BLE001
            out["skipped"].append({"digest": raw.get("digest"), "error": str(exc)})
            continue
        if entry is None:
            out["skipped"].append({"digest": raw.get("digest"),
                                   "error": "no charter axis or harm scored"})
            continue
        batch = store.add(address, int(epoch), axis_ids, entry)
        out["queued"] += 1
    out["epoch"] = int(batch.epoch) if batch is not None else int(epoch)
    return out
