"""Reviews on ToolBond (Decision 2026-11-09, docs/tool_substrate.md).

Every review of a tool is one entry in its agent's batch for the epoch: the
four fixed fields (correctness, simplicity, novelty, malware), the charter
axes the reviewer scored, and an evidence hash. The agent commits the batch
once per epoch (a hash and the review count; no ATN stake) and reveals it
after the epoch is anchored, within ToolBond's reveal window. ToolBond
settles each tool's round on chain: the REP-weighted sums per field and
axis, each reviewer's accuracy over all of them, and REP moved by accuracy
(trustless-contracts ToolBond f633bc1). A household with no REP cannot
commit (ToolBond reverts NoWeight), so the daemon queues nothing for it.

A reveal is relayable: anyone holding the plaintext and the salt can submit
it, so the batch is kept on disk as a sealed reveal the owner can hand to a
backup. A batch never revealed loses REP for each review it declared, so
revealing always beats withholding.

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
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

NO_SCORE = -128          # int8 sentinel: no score on this charter axis
MAX_AXES = 16

# ToolBond.ReviewInput: digest, correctness (-100..100), simplicity (0..100),
# novelty (0..100), malware (0 or 1), charter scores, evidence hash.
REVIEW_INPUT_TYPE = "(bytes32,int8,uint8,uint8,uint8,bytes,bytes32)"

# Counts every write to the batch store and the held reviews in this
# process, so a cache of what they hold (tool_search.household_reviews,
# behind review_due) knows when to read them again. A directory's mtime
# can't tell: exFAT and FAT never update it when a file inside changes, and
# on any filesystem two writes within its timestamp resolution look alike.
_writes = 0
_writes_lock = threading.Lock()


def writes() -> int:
    """How many batch or held-review writes this process has made."""
    return _writes


def _wrote() -> None:
    global _writes
    with _writes_lock:
        _writes += 1


def _fn(name: str, inputs: list, outputs: list = (), mutability: str = "nonpayable") -> dict[str, Any]:
    def arg(n: str, t: str) -> dict[str, Any]:
        if t == "ReviewInput[]":
            return {"name": n, "type": "tuple[]", "components": [
                {"name": "digest", "type": "bytes32"},
                {"name": "correctness", "type": "int8"},
                {"name": "simplicity", "type": "uint8"},
                {"name": "novelty", "type": "uint8"},
                {"name": "malware", "type": "uint8"},
                {"name": "scores", "type": "bytes"},
                {"name": "evidenceHash", "type": "bytes32"}]}
        return {"name": n, "type": t}
    return {"name": name, "type": "function", "stateMutability": mutability,
            "inputs": [arg(n, t) for n, t in inputs],
            "outputs": [{"name": n, "type": t} for n, t in outputs]}


REVIEW_TUPLE = [
    ("reviewer", "address"), ("correctness", "int8"), ("simplicity", "uint8"),
    ("novelty", "uint8"), ("malware", "uint8"), ("claimed", "bool"),
    ("bountied", "bool"), ("bondClaimed", "bool"), ("accuracy", "uint64"),
    ("weight", "uint128"), ("scores", "bytes16"),
]


def _review_fn(name: str, inputs: list) -> dict[str, Any]:
    return {"name": name, "type": "function", "stateMutability": "view",
            "inputs": [{"name": n, "type": t} for n, t in inputs],
            "outputs": [{"name": "", "type": "tuple", "components": [
                {"name": n, "type": t} for n, t in REVIEW_TUPLE]}]}


# ToolBond.Stage
STAGE_NONE, STAGE_OPEN, STAGE_FINALIZING, STAGE_DONE = range(4)
STAGES = ("none", "open", "finalizing", "done")
# Review.accuracy when nothing could be scored (no gain or loss).
UNINFORMATIVE = 2 ** 64 - 1

TOOL_BOND_REVIEW_ABI = [
    _fn("currentEpoch", [], [("", "uint256")], "view"),
    _fn("epochInfo", [("epoch", "uint256")],
        [("started", "bool"), ("charterHash", "bytes32"), ("axisIds", "bytes32[]"),
         ("snapshot", "uint48"), ("supply", "uint256"), ("revealDeadline", "uint256"),
         ("unscoredBps", "uint256")], "view"),
    _fn("getRound", [("digest", "bytes32"), ("epoch", "uint256")],
        [("stage", "uint8"), ("weight", "uint256"), ("reviewCount", "uint256"),
         ("beforeConclusion", "bool")], "view"),
    _fn("roundAxis", [("digest", "bytes32"), ("epoch", "uint256"), ("i", "uint256")],
        [("sum", "int256"), ("mass", "uint256")], "view"),
    _review_fn("reviewAt", [("digest", "bytes32"), ("epoch", "uint256"), ("i", "uint256")]),
    _review_fn("reviewOf", [("digest", "bytes32"), ("epoch", "uint256"), ("reviewer", "address")]),
    _fn("isBonded", [("digest", "bytes32")], [("", "bool")], "view"),
    _fn("reviewerShareBps", [], [("", "uint256")], "view"),
    _fn("revealEnds", [("epoch", "uint256")], [("", "uint256")], "view"),
    _fn("rep", [], [("", "address")], "view"),
    _fn("commits", [("", "uint256"), ("", "address")],
        [("commitment", "bytes32"), ("weight", "uint128"),
         ("revealed", "bool"), ("forfeited", "bool")], "view"),
    _fn("commitmentOf", [("epoch", "uint256"), ("agent", "address"),
                         ("reviews", "ReviewInput[]"), ("salt", "bytes32")],
        [("", "bytes32")], "pure"),
    # The commit declares its review count (public); the reveal must hold
    # exactly that many. No ATN rides on it.
    _fn("commit", [("commitment", "bytes32"), ("reviews", "uint256")]),
    _fn("reveal", [("epoch", "uint256"), ("agent", "address"),
                   ("reviews", "ReviewInput[]"), ("salt", "bytes32")]),
    _fn("forfeit", [("epoch", "uint256"), ("agent", "address")]),
    _fn("finalize", [("digest", "bytes32"), ("epoch", "uint256"), ("maxReviews", "uint256")]),
    _fn("claim", [("digest", "bytes32"), ("epoch", "uint256"), ("reviewer", "address")]),
    _fn("toolRating", [("digest", "bytes32"), ("key", "bytes32")],
        [("sum", "int256"), ("mass", "uint256")], "view"),
]

# Substrate's owner binding and RepToken's balance: a household's REP.
_HOUSEHOLD_ABI = [
    _fn("getAgentOwner", [("agent", "address")], [("", "address")], "view"),
]
_REP_BALANCE_ABI = [
    _fn("balanceOf", [("account", "address")], [("", "uint256")], "view"),
]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_scores(axes: dict[str, float] | None, axis_ids: list[str]) -> bytes:
    """One int8 per charter axis, in charter order: round(score x 100),
    clamped to [-100, 100]; NO_SCORE where the review has no score. Empty
    bytes when the review scores no axis at all (ToolBond reads that as
    every axis unscored)."""
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


def _hundredths(value: float, lo: int) -> int:
    return max(lo, min(100, int(round(float(value) * 100))))


def _b32(h: str) -> bytes:
    s = str(h or "").lower().removeprefix("0x")
    if len(s) != 64:
        raise ValueError(f"expected a 32-byte hex value, got {h!r}")
    return bytes.fromhex(s)


@dataclass
class ReviewEntry:
    """One review in an agent's epoch batch (Decision 2026-11-09)."""
    digest: str                      # tool manifest digest (hex)
    correctness: float = 0.0         # -1..1
    simplicity: float = 0.0          # 0..1
    novelty: float = 0.0             # 0..1
    malware: bool = False
    axes: dict[str, float] = field(default_factory=dict)   # charter axis id -> [-1, 1]
    evidence_hash: str = "00" * 32   # hex; sha256 of the evidence blob

    def to_input(self, axis_ids: list[str]) -> tuple:
        return (_b32(self.digest), _hundredths(self.correctness, -100),
                _hundredths(self.simplicity, 0), _hundredths(self.novelty, 0),
                1 if self.malware else 0, encode_scores(self.axes, axis_ids),
                _b32(self.evidence_hash))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Optional["ReviewEntry"]:
        """A stored entry, or None when it lacks a fixed field (a batch
        written before the fields existed)."""
        known = {k: d[k] for k in ("digest", "correctness", "simplicity",
                                   "novelty", "malware", "axes", "evidence_hash")
                 if k in d}
        if not all(k in known for k in ("digest", "correctness", "simplicity",
                                        "novelty", "malware")):
            return None
        return cls(**known)


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
        entries = [ReviewEntry.from_dict(r) for r in d.get("reviews", [])
                   if isinstance(r, dict)]
        return EpochBatch(epoch=int(d["epoch"]), agent=d["agent"], axis_ids=list(d["axis_ids"]),
                          reviews=[e for e in entries if e is not None],
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
        _wrote()

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


# Batch states whose reviews never count: forfeited unrevealed, rolled into
# a later batch, or refused (the household held no REP).
DEAD_STATES = ("forfeited", "rolled", "refused")


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
            c, weight, revealed, forfeited = b.functions.commits(
                int(epoch), Web3.to_checksum_address(agent)).call()
            return {"commitment": c.hex(), "weight": int(weight),
                    "revealed": bool(revealed), "forfeited": bool(forfeited)}
        return await self._read(fn)

    async def round_of(self, digest: str, epoch: int) -> Optional[dict[str, Any]]:
        def fn(b, s):
            r = b.functions.getRound(_b32(digest), int(epoch)).call()
            return {"stage": int(r[0]), "weight": int(r[1]), "reviews": int(r[2]),
                    "before_conclusion": bool(r[3])}
        return await self._read(fn)

    async def review_of(self, digest: str, epoch: int,
                        reviewer: str) -> Optional[dict[str, Any]]:
        def fn(b, s):
            from web3 import Web3
            v = b.functions.reviewOf(_b32(digest), int(epoch),
                                     Web3.to_checksum_address(reviewer)).call()
            return {"reviewer": v[0], "correctness": int(v[1]), "simplicity": int(v[2]),
                    "novelty": int(v[3]), "malware": int(v[4]), "claimed": bool(v[5]),
                    "accuracy": int(v[8]), "weight": int(v[9])}
        return await self._read(fn)

    async def household_rep(self, agent: str) -> Optional[int]:
        """The REP of ``agent``'s household (its bound owner, else itself),
        as ToolBond weighs a commit; None when it can't be read."""
        def run() -> int:
            from web3 import Web3
            w3 = self._base._w3()
            bond, substrate = self._contracts(w3)
            sub = w3.eth.contract(address=substrate.address, abi=_HOUSEHOLD_ABI)
            a = Web3.to_checksum_address(agent)
            owner = sub.functions.getAgentOwner(a).call()
            house = a if int(owner, 16) == 0 else Web3.to_checksum_address(owner)
            rep = w3.eth.contract(address=bond.functions.rep().call(), abi=_REP_BALANCE_ABI)
            return int(rep.functions.balanceOf(house).call())
        try:
            return await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            log.debug("household REP read failed: %s", exc)
            return None

    async def commit(self, private_key: str, batch: EpochBatch) -> dict[str, Any]:
        """Seal and commit ``batch``. The salt is fresh per batch; the
        caller saves the batch so the reveal survives a restart."""
        from eth_account import Account
        agent = Account.from_key(private_key).address
        if agent.lower() != batch.agent.lower():
            return {"success": False, "error": "batch belongs to another agent"}
        if not batch.salt:
            batch.salt = secrets.token_hex(32)
        c = commitment_of(batch.epoch, agent, batch.inputs(), bytes.fromhex(batch.salt))
        batch.commitment = c.hex()
        count = len(batch.inputs())
        res = await self._write(private_key, lambda b, s, a: b.functions.commit(
                                    c, int(count)),
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


# Reviews from an agent that is not registered yet are held, not dropped
# (Decision 2026-11-02, docs/tool_codebases.md): they are submitted to the
# open round once the household registers.
HELD_DIR = "held_reviews"
HELD_MAX = 500
HELD_NOTE = (
    "held: reviews go on chain from a registered agent. This one is kept "
    "and submitted to the open round once your household is registered "
    "(the owner connects a wallet in the Autonet app and approves your "
    "registration). Your reviews weigh by your household's REP, and a "
    "household without REP cannot review.")
# An external agent's reviews go out from the operator (Decision 2026-11-08).
HELD_NOTE_OPERATOR = (
    "held: you review as this household's operator, which is not registered "
    "yet. This review is kept and submitted to the open round once it is "
    "(the owner registers the operator in the Autonet app, Household). Your "
    "reviews weigh by your household's REP, and a household without REP "
    "cannot review.")
NO_REP_NOTE = (
    "not sent: your household holds no REP, and ToolBond refuses reviews "
    "from a household without it (Decision 2026-11-09). REP comes from the "
    "tool pool, as an author of tools other households review.")


def note_household_rep(runtime: Any, rep: Optional[int]) -> None:
    """Remember this daemon's household REP (one household per daemon):
    ``review_due`` stays quiet while it is 0. None = unknown, ignored."""
    if rep is None:
        return
    try:
        runtime.__dict__["_household_rep"] = int(rep)
    except Exception:  # noqa: BLE001
        pass


def household_has_no_rep(runtime: Any) -> bool:
    """True once a read showed the household holds no REP."""
    try:
        return runtime.__dict__.get("_household_rep") == 0
    except Exception:  # noqa: BLE001
        return False


def all_held_reviews(runtime: Any) -> list[dict[str, Any]]:
    """The reviews held for every agent on this daemon."""
    root = Path(runtime._config.data_dir) / HELD_DIR
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 (unreadable: nothing held there)
            continue
        held = data.get("reviews") if isinstance(data, dict) else None
        if isinstance(held, list):
            out.extend(r for r in held if isinstance(r, dict))
    return out


def _held_path(runtime: Any, agent_id: str) -> Path:
    import re
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", agent_id)
    return Path(runtime._config.data_dir) / HELD_DIR / f"{safe}.json"


def held_reviews(runtime: Any, agent_id: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(_held_path(runtime, agent_id).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 (none held, or unreadable)
        return []
    held = data.get("reviews") if isinstance(data, dict) else None
    return held if isinstance(held, list) else []


def hold_reviews(runtime: Any, agent_id: str,
                 reviews: list[dict[str, Any]]) -> int:
    """Keep ``reviews`` for ``agent_id`` until it registers (newest
    ``HELD_MAX``). Returns how many are held now."""
    held = held_reviews(runtime, agent_id)
    now = int(time.time())
    held.extend({**dict(r), "held_ts": now} for r in reviews)
    held = held[-HELD_MAX:]
    path = _held_path(runtime, agent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"agent_id": agent_id, "reviews": held}),
                   encoding="utf-8")
    os.replace(tmp, path)
    _wrote()
    return len(held)


async def flush_held_reviews(runtime: Any) -> int:
    """Submit held reviews of agents that have registered since. Returns how
    many were queued. Never raises."""
    root = Path(runtime._config.data_dir) / HELD_DIR
    if not root.is_dir():
        return 0
    queued = 0
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(data, dict) or not data.get("reviews"):
            path.unlink(missing_ok=True)
            _wrote()
            continue
        agent_id = str(data.get("agent_id") or "")
        defn = runtime.get_agent(agent_id) if agent_id else None
        if defn is None:
            continue
        data = data["reviews"]
        identity = getattr(defn, "identity", None)
        if not getattr(identity, "registered_on_chain", False):
            continue
        reviews = [{k: v for k, v in r.items() if k != "held_ts"} for r in data]
        out = await queue_reviews(runtime, agent_id, reviews)
        if out.get("epoch") is not None:
            queued += int(out.get("queued") or 0)
            path.unlink(missing_ok=True)
            _wrote()
            log.info("submitted %d held review(s) of %s", out.get("queued"), agent_id)
    return queued


def review_error(raw: dict[str, Any]) -> str:
    """Why ``raw`` is not a valid review, or "" (Decision 2026-11-09): every
    review carries correctness, simplicity, novelty and malware, and a
    malware flag carries evidence (its blob's sha256 as ``evidence_hash``)."""
    from nodes.common.library.review_fields import parse_fields
    fields, why = parse_fields(raw)
    if fields is None:
        return why
    if fields["malware"] and not _evidence_hash(raw.get("evidence_hash")):
        return ("a malware flag needs evidence: say what the code does and "
                "point into it (file and line numbers, or the quoted code)")
    return ""


def _evidence_hash(value: Any) -> str:
    h = str(value or "").lower().removeprefix("0x")
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h) or h == "0" * 64:
        return ""
    return h


def normalize_review(raw: dict[str, Any], axis_ids: list[str]) -> Optional[ReviewEntry]:
    """A review step's {digest, correctness, simplicity, novelty, malware,
    axes?, evidence_hash?} as a ReviewEntry, or None when it is not a valid
    review (``review_error``). Charter axes outside the charter are dropped;
    scores clamp to their ranges."""
    from nodes.common.library.review_fields import parse_fields
    if review_error(raw):
        return None
    fields, _ = parse_fields(raw)
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
    return ReviewEntry(digest=str(raw["digest"]).lower().removeprefix("0x"),
                       correctness=fields["correctness"],
                       simplicity=fields["simplicity"], novelty=fields["novelty"],
                       malware=fields["malware"], axes=dict(sorted(axes.items())),
                       evidence_hash=_evidence_hash(raw.get("evidence_hash")) or "00" * 32)


async def queue_reviews(runtime: Any, agent_id: str,
                        reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Queue the reviewing agent's reviews into its batch for ToolBond's
    current epoch. Never raises; returns {queued, epoch, skipped, note}.
    Each review is {digest, correctness, simplicity, novelty, malware,
    axes?, evidence_hash?, author?}; ``author`` (the manifest author) lets
    the agent's own tools be skipped. An invalid review (a fixed field
    missing, a malware flag without evidence) is skipped with the reason.

    The agent must be registered on chain with a daemon-held key (the batch
    is committed with its key), and ToolBond must be reachable to read the
    epoch. A household holding no REP gets nothing queued: ToolBond refuses
    its commit.

    An external agent's identity is the operator's (Decision 2026-11-08),
    so its reviews land in the operator's batch: every external session of
    the household fills one batch per epoch, and a second review of a tool
    replaces the first (``ReviewBatchStore.add``)."""
    from nodes.common.library.charter import active_charter

    out: dict[str, Any] = {"queued": 0, "epoch": None, "skipped": []}
    if not reviews:
        return out
    defn = runtime.get_agent(agent_id) if agent_id else None
    identity = getattr(defn, "identity", None) if defn else None
    address = str(getattr(identity, "address", "") or "")
    check = getattr(getattr(runtime, "registry", None), "is_operator_identity", None)
    as_operator = bool(check(identity)) if callable(check) and identity is not None else False
    if as_operator:
        out["reviewer"] = address
    if not address or not getattr(identity, "registered_on_chain", False):
        if defn is None:
            out["note"] = "no such agent; nothing was queued"
            return out
        out["held"] = hold_reviews(runtime, agent_id, reviews)
        out["note"] = HELD_NOTE_OPERATOR if as_operator else HELD_NOTE
        return out
    client = client_for(runtime)
    if not client.available():
        out["note"] = "no ToolBond configured; reviews were not queued"
        return out
    epoch = await client.current_epoch()
    if epoch is None:
        out["note"] = "ToolBond unreachable; reviews were not queued"
        return out
    rep = await client.household_rep(address) if hasattr(client, "household_rep") else None
    note_household_rep(runtime, rep)
    if rep == 0:
        out["note"] = NO_REP_NOTE
        out["skipped"] = [{"digest": r.get("digest"), "error": "household holds no REP"}
                          for r in reviews]
        return out
    axis_ids = list(active_charter().ids)
    store = store_for(runtime)
    batch = None
    for raw in reviews:
        if str(raw.get("author") or "").lower() == address.lower():
            # ToolBond never counts an author's review of their own tool.
            out["skipped"].append({"digest": raw.get("digest"),
                                   "error": "you cannot review your own tool"})
            continue
        try:
            why = review_error(raw)
            entry = None if why else normalize_review(raw, axis_ids)
        except Exception as exc:  # noqa: BLE001
            out["skipped"].append({"digest": raw.get("digest"), "error": str(exc)})
            continue
        if entry is None:
            out["skipped"].append({"digest": raw.get("digest"),
                                   "error": why or "not a valid review"})
            continue
        batch = store.add(address, int(epoch), axis_ids, entry)
        out["queued"] += 1
    out["epoch"] = int(batch.epoch) if batch is not None else int(epoch)
    return out
