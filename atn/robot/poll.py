"""The poll rail (docs/robots.md, Decision 2026-10-25, Phase A).

Pure functions, no I/O:

- **Who polls.** Each round, each robot gets its own ``k`` pollers, drawn
  REP-weighted from the agents of households at or above a REP percentile
  (at least ``k`` households stay eligible). The seed is an anchored epoch
  root from two anchors before the round's own, plus the robot's address,
  so the draw is public and anyone can recompute it. At most one poller
  per household; the robot's own household and sanctioned households never
  poll. Integer arithmetic only, so every daemon draws the same pollers.
- **When they poll.** A poller signs ``(round, seed, robot, slot)``.
  Signatures are deterministic and low-s only (``crypto``), so that one
  signature fixes both the poll time inside the round and the challenge.
  Only the poller can compute either in advance; afterwards anyone can
  check neither was cherry-picked.
- **Receipts, aggregates, reports.** The poller countersigns each answer
  (a receipt, archived by the robot), bundles its round's receipts into one
  signed aggregate for gossip, and signs a report for every exception.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import crypto as C
from . import worklog as W

POLL_TYPE = "autonet.robot.poll/1"
RECEIPT_TYPE = "autonet.robot.receipt/1"
AGGREGATE_TYPE = "autonet.robot.aggregate/1"
REPORT_TYPE = "autonet.robot.report/1"

WRONG_PACKS = "active packs differ from the required set"

# Exceptions a report can name.
REPORT_KINDS = ("no_answer", "late", "invalid", "chain_break", "wrong_packs",
                "missing_commit", "poller_absent", "challenge_missing", "scene_conflict")


@dataclass(frozen=True)
class Member:
    agent: str       # agent address, lowercased
    household: str   # owner wallet (or the agent itself when unbound)
    rep: int         # REP at the round's snapshot, base units


@dataclass(frozen=True)
class Round:
    """One poll round, ``index``, from ``start`` for ``period_s``. ``seed``
    is the epoch root of the anchor two before the one in force at its start.
    ``standards`` are the safety-pack digests every robot must run this round
    (its jurisdiction's and every ancestor's); empty means none are set."""
    index: int
    start: float
    seed: str
    period_s: float
    members: tuple[Member, ...]
    sanctioned: frozenset[str] = field(default_factory=frozenset)
    standards: tuple[str, ...] = ()

    def household_of(self, agent: str) -> str:
        agent = agent.lower()
        for m in self.members:
            if m.agent == agent:
                return m.household
        return agent


def eligible_households(members: Iterable[Member], *, percentile: int, k: int,
                        exclude: Iterable[str] = ()) -> set[str]:
    """Households at or above the REP ``percentile``, widened down the
    ranking until at least ``k`` are eligible (ties at the cut all stay)."""
    excluded = {h.lower() for h in exclude}
    totals: dict[str, int] = {}
    for m in members:
        if m.household not in excluded:
            totals[m.household] = totals.get(m.household, 0) + m.rep
    ranked = sorted(((r, h) for h, r in totals.items() if r > 0),
                    key=lambda x: (-x[0], x[1]))
    if not ranked:
        return set()
    n = len(ranked)
    top = -(-n * (100 - percentile) // 100)    # ceil, integers only
    keep = min(n, max(k, top, 1))
    cut = ranked[keep - 1][0]
    return {h for r, h in ranked if r >= cut}


def draw_pollers(rnd: Round, robot: str, *, k: int, percentile: int) -> list[str]:
    """The robot's pollers for the round, in slot order."""
    robot = robot.lower()
    houses = eligible_households(
        rnd.members, percentile=percentile, k=k,
        exclude={rnd.household_of(robot), *rnd.sanctioned})
    cands = sorted((m for m in rnd.members if m.household in houses and m.rep > 0),
                   key=lambda m: m.agent)
    seed = bytes.fromhex(rnd.seed.removeprefix("0x"))
    chosen: list[str] = []
    j = 0
    while len(chosen) < k and cands:
        total = sum(m.rep for m in cands)
        h = hashlib.sha256(b"autonet:robot-poller:v1" + seed + robot.encode()
                           + j.to_bytes(4, "big")).digest()
        r = int.from_bytes(h, "big") % total
        for m in cands:
            if r < m.rep:
                break
            r -= m.rep
        chosen.append(m.agent)
        cands = [c for c in cands if c.household != m.household]
        j += 1
    return chosen


# -- the poll ---------------------------------------------------------------

def poll_body(rnd_index: int, seed: str, robot: str, slot: int) -> bytes:
    return f"{POLL_TYPE}|{rnd_index}|{seed.lower()}|{robot.lower()}|{slot}".encode()


def make_poll(rnd: Round, robot: str, slot: int, *, poller: str,
              private_key: str) -> dict[str, Any]:
    return {"type": POLL_TYPE, "round": rnd.index, "seed": rnd.seed.lower(),
            "robot": robot.lower(), "slot": slot, "poller": poller.lower(),
            "sig": C.sign_bytes(poll_body(rnd.index, rnd.seed, robot, slot), private_key)}


def poll_id(poll: dict[str, Any]) -> str:
    return C.sha256_hex(bytes.fromhex(poll["sig"].removeprefix("0x")))


def poll_timing(poll: dict[str, Any], rnd: Round) -> tuple[float, str]:
    """(when the poll is due, the challenge), both fixed by the signature."""
    sig = bytes.fromhex(poll["sig"].removeprefix("0x"))
    challenge = hashlib.sha256(b"autonet:robot-challenge:v1" + sig).hexdigest()
    span_ms = max(1, int(rnd.period_s * 1000))
    offset_ms = int.from_bytes(hashlib.sha256(b"autonet:robot-at:v1" + sig).digest(),
                               "big") % span_ms
    return rnd.start + offset_ms / 1000.0, challenge


def verify_poll(poll: dict[str, Any], rnd: Round, *, k: int, percentile: int,
                now: float | None = None, early_slack_s: float = 2.0) -> list[str]:
    """Why a received poll is not valid (empty = valid). A poll that
    arrives before its due time is refused: the time is the poller's to
    keep secret, not to choose."""
    errors = []
    if poll.get("type") != POLL_TYPE:
        return ["wrong poll type"]
    if poll.get("round") != rnd.index or str(poll.get("seed", "")).lower() != rnd.seed.lower():
        return ["poll is for a different round"]
    robot, slot = str(poll.get("robot", "")), poll.get("slot")
    try:
        who = C.recover_bytes(poll_body(rnd.index, rnd.seed, robot, slot), poll.get("sig", ""))
    except Exception as exc:  # noqa: BLE001
        return [f"bad signature: {exc}"]
    if who != str(poll.get("poller", "")).lower():
        errors.append("signed by a different key")
    drawn = draw_pollers(rnd, robot, k=k, percentile=percentile)
    if not isinstance(slot, int) or not 0 <= slot < len(drawn) or drawn[slot] != who:
        errors.append("poller was not drawn for this robot and slot")
    at, _ = poll_timing(poll, rnd)
    now = time.time() if now is None else now
    if now < at - early_slack_s:
        errors.append(f"early: due at {at:.1f}, now {now:.1f}")
    if now > rnd.start + rnd.period_s + early_slack_s:
        errors.append("round is over")
    return errors


def verify_answer(links: list[dict[str, Any]], poll: dict[str, Any], rnd: Round, *,
                  answerer: str, known_head: dict[str, Any] | None,
                  sent_at: float, received_at: float, deadline_s: float,
                  required_packs: list[str] | None = None) -> list[str]:
    """The poller's check of an answer: the links since the head it already
    holds, ending in a link that carries this poll's challenge. Timeliness
    is judged by the poller's OWN clock, never the robot's ``ts``."""
    errors = []
    if not links:
        return ["empty answer"]
    chain = ([known_head] if known_head else []) + list(links)
    errors += W.verify_chain(chain, robot=answerer)
    last = links[-1]
    _, challenge = poll_timing(poll, rnd)
    if last.get("challenge") != challenge:
        errors.append("challenge mismatch")
    if last.get("poll") != poll_id(poll):
        errors.append("answer names a different poll")
    if received_at - sent_at > deadline_s:
        errors.append(f"late: {received_at - sent_at:.2f}s > {deadline_s:.2f}s")
    if required_packs is not None and sorted(last.get("packs") or []) != sorted(required_packs):
        errors.append(WRONG_PACKS)
    return errors


# -- receipts, aggregates, reports -----------------------------------------

def make_receipt(poll: dict[str, Any], link: dict[str, Any], *, role: str,
                 received_at: float, poller: str, private_key: str) -> dict[str, Any]:
    """The poller's countersignature on one answer. ``role`` is ``subject``
    for the polled robot, ``witness`` for a robot answering as its witness."""
    return C.sign({"type": RECEIPT_TYPE, "round": poll["round"], "poll": poll_id(poll),
                   "subject": poll["robot"], "robot": link["robot"], "role": role,
                   "link": C.obj_hash(link), "seq": link["seq"],
                   "received_at": received_at, "poller": poller.lower()}, private_key)


def verify_receipt(receipt: dict[str, Any], link: dict[str, Any]) -> list[str]:
    errors = []
    if receipt.get("type") != RECEIPT_TYPE:
        errors.append("wrong receipt type")
    try:
        if C.signer(receipt) != str(receipt.get("poller", "")).lower():
            errors.append("receipt not signed by its poller")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"bad signature: {exc}")
    if receipt.get("link") != C.obj_hash(link) or receipt.get("seq") != link.get("seq"):
        errors.append("receipt is for a different link")
    return errors


def make_aggregate(rnd_index: int, receipts: list[dict[str, Any]], *, poller: str,
                   private_key: str) -> dict[str, Any]:
    hashes = [C.obj_hash(r) for r in receipts]
    return C.sign({"type": AGGREGATE_TYPE, "round": rnd_index, "poller": poller.lower(),
                   "receipts": hashes, "root": C.merkle_root(hashes)}, private_key)


def receipt_in_aggregate(receipt: dict[str, Any], agg: dict[str, Any]) -> bool:
    try:
        if C.signer(agg) != str(agg.get("poller", "")).lower():
            return False
    except Exception:  # noqa: BLE001
        return False
    hashes = agg.get("receipts") or []
    return (C.merkle_root(hashes) == agg.get("root")
            and C.obj_hash(receipt) in hashes)


def make_report(kind: str, *, reporter: str, private_key: str,
                now: float | None = None, **detail: Any) -> dict[str, Any]:
    if kind not in REPORT_KINDS:
        raise ValueError(f"unknown report kind {kind!r}")
    return C.sign({"type": REPORT_TYPE, "kind": kind, "reporter": reporter.lower(),
                   "ts": time.time() if now is None else now, **detail}, private_key)
