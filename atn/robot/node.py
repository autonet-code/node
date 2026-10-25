"""One daemon's part in the poll rail (docs/robots.md, 2026-10-25).

A :class:`PollNode` plays every role the daemon has:

- **robot** (it has a :class:`~atn.robot.worklog.Worklog`): commits a link
  each time chunks close and pushes it to its witnesses and current
  pollers; answers polls; archives the receipts; at a round's end reports
  drawn pollers that never polled it.
- **witness** (it is in a polled robot's roster): answers the same poll
  from its own worklog, so every log carries the challenge.
- **poller** (it was drawn): polls at the time its signature fixes, sends
  to the robot and its witnesses at once, verifies and countersigns, and
  gossips one aggregate per round.
- **watcher** (everyone holding a robot's head): reports a broken chain or
  a commitment that stops arriving.

Transport is injected: request/response to one peer, push to one peer,
gossip to all. :class:`LocalNetwork` is the in-process transport the tests
and the sim E2E use.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from . import poll as P
from . import worklog as W

log = logging.getLogger(__name__)


@dataclass
class PollParams:
    k: int = 3                 # pollers per robot per round (provisional)
    percentile: int = 90       # REP floor, relative (provisional)
    deadline_s: float = 2.0    # answer deadline, poller's clock
    commit_gap_s: float = 30.0  # a robot silent this long is reported
    early_slack_s: float = 2.0  # clock skew allowed on a poll's due time
    tick_s: float = 0.5


class Transport(Protocol):
    async def request(self, to: str, msg: dict, timeout: float) -> dict: ...
    async def push(self, to: str, msg: dict) -> None: ...
    async def gossip(self, msg: dict) -> None: ...


class PollNode:
    def __init__(self, *, address: str, private_key: str, transport: Transport,
                 params: PollParams, round_at: Callable[[float], P.Round | None],
                 roster: Callable[[str], list[str]], robots: Callable[[], list[str]],
                 state_dir: str | Path, worklog: W.Worklog | None = None,
                 packs: Callable[[], list[str]] = list,
                 write_challenge: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.address = address.lower()
        self._key = private_key
        self.transport = transport
        self.params = params
        self.round_at = round_at          # the round in force at a time
        self.roster = roster              # robot -> its witnesses
        self.robots = robots              # every robot this node may poll
        self.worklog = worklog
        self.packs = packs
        self.write_challenge = write_challenge
        self.clock = clock
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # watcher state: robot -> (head link, local time it arrived)
        self.heads: dict[str, tuple[dict, float]] = {}
        self._gap_reported: set[tuple[str, int]] = set()
        self._watch_from: dict[str, float] = {}
        self.missed: dict[str, int] = {}    # robot -> links skipped between pushes
        # robot state
        self.receipts: list[dict] = self._load("receipts.jsonl")
        self._polled_by: dict[int, set[str]] = {}
        self._absent_done: set[int] = set()
        # poller state
        self._done: set[tuple[int, str, int]] = set()
        self._my_receipts: dict[int, list[dict]] = {}
        self._aggregated: set[int] = set()
        # everything seen on gossip (aggregates, reports)
        self.gossip_log: list[dict] = []
        self.reports_sent: list[dict] = []
        self._draws: dict[tuple, list[str]] = {}
        self._pending_push: list[dict] = []     # poll-answer links not yet pushed
        self._task: asyncio.Task | None = None

    def _draw(self, rnd: P.Round, robot: str) -> list[str]:
        """The robot's pollers for the round, memoized (a round's draw never
        changes)."""
        key = (rnd.index, rnd.seed, robot.lower())
        drawn = self._draws.get(key)
        if drawn is None:
            if len(self._draws) > 4096:
                self._draws.clear()
            drawn = self._draws[key] = P.draw_pollers(
                rnd, robot, k=self.params.k, percentile=self.params.percentile)
        return drawn

    # -- persistence ---------------------------------------------------------

    def _load(self, name: str) -> list[dict]:
        p = self.state_dir / name
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]

    def _append(self, name: str, obj: dict) -> None:
        with (self.state_dir / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, sort_keys=True) + "\n")

    # -- inbound ---------------------------------------------------------------

    async def handle_request(self, msg: dict) -> dict:
        if msg.get("type") == "poll":
            ans = self._answer(msg["poll"], int(msg.get("since", -1)))
            if ans.get("links"):
                # The answer's link goes to every watcher too, not only to
                # this poller, so their heads and chain checks stay current.
                self._pending_push.append(ans["links"][-1])
            return ans
        return {"error": f"unknown request {msg.get('type')!r}"}

    async def handle_push(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == W.LINK_TYPE:
            await self._on_link(msg)
        elif kind == P.RECEIPT_TYPE:
            self._on_receipt(msg)

    async def handle_gossip(self, msg: dict) -> None:
        self.gossip_log.append(msg)

    # -- robot / witness: answering --------------------------------------------

    def _answer(self, poll: dict, since: int) -> dict:
        if self.worklog is None:
            return {"error": "no worklog on this daemon"}
        rnd = self._round(poll.get("round"))
        if rnd is None:
            return {"error": "poll is not for the current round"}
        errs = P.verify_poll(poll, rnd, k=self.params.k, percentile=self.params.percentile,
                             now=self.clock(), early_slack_s=self.params.early_slack_s)
        subject = str(poll.get("robot", "")).lower()
        if subject != self.address and self.address not in self.roster(subject):
            errs.append("this daemon is not a witness of the polled robot")
        if errs:
            return {"error": "; ".join(errs)}
        _, challenge = P.poll_timing(poll, rnd)
        if self.write_challenge is not None:
            self.write_challenge(challenge)
        if subject == self.address:
            self._polled_by.setdefault(rnd.index, set()).add(poll["poller"])
        link = self.worklog.commit(packs=self.packs(), challenge=challenge,
                                   poll=P.poll_id(poll), force=True, now=self.clock())
        links = self.worklog.links_since(since) if since >= 0 else [link]
        return {"links": links}

    def _on_receipt(self, receipt: dict) -> None:
        if self.worklog is None or receipt.get("robot") != self.address:
            return
        link = next((lk for lk in self.worklog.links if lk["seq"] == receipt.get("seq")), None)
        if link is None or P.verify_receipt(receipt, link):
            log.warning("robot: dropped a receipt that does not match our chain")
            return
        self.receipts.append(receipt)
        self._append("receipts.jsonl", receipt)

    # -- watcher ---------------------------------------------------------------

    async def _on_link(self, link: dict) -> None:
        robot = str(link.get("robot", "")).lower()
        held = self.heads.get(robot)
        prev = held[0] if held else None
        if prev is not None and link.get("seq", -1) <= prev["seq"]:
            return  # old or duplicate
        follows = prev is not None and link.get("seq") == prev["seq"] + 1
        if prev is not None and not follows:
            # Links we never saw; chain continuity across them goes unchecked.
            self.missed[robot] = self.missed.get(robot, 0) + link["seq"] - prev["seq"] - 1
        errs = W.verify_link(link, robot=robot, prev=prev if follows else None)
        if errs:
            await self._report("chain_break" if follows else "invalid", robot=robot,
                               seq=link.get("seq"), errors=errs)
            return
        self.heads[robot] = (link, self.clock())

    def _watching(self) -> set[str]:
        """Robots that push their links here right now: those we witness
        and those we were drawn to poll this round."""
        rnd = self.round_at(self.clock())
        out = set()
        for robot in self.robots():
            robot = robot.lower()
            if robot == self.address:
                continue
            if self.address in self.roster(robot) or (rnd is not None and self.address in
                                                      self._draw(rnd, robot)):
                out.add(robot)
        return out

    async def _watch(self) -> None:
        """Report a watched robot whose commitments stopped arriving. Silence
        is counted from when we started watching it, at the earliest."""
        now = self.clock()
        watching = self._watching()
        for robot in watching - set(self._watch_from):
            self._watch_from[robot] = now
        for robot in set(self._watch_from) - watching:
            del self._watch_from[robot]
        for robot in watching:
            head, seen = self.heads.get(robot, (None, 0.0))
            seen = max(seen, self._watch_from[robot])
            seq = head["seq"] if head else -1
            if now - seen > self.params.commit_gap_s and (robot, seq) not in self._gap_reported:
                self._gap_reported.add((robot, seq))
                await self._report("missing_commit", robot=robot, last_seq=seq,
                                   silent_s=round(now - seen, 1))

    # -- robot: continuous commitment --------------------------------------------

    async def _commit_and_push(self) -> None:
        if self.worklog is None:
            return
        link = self.worklog.commit(packs=self.packs(), now=self.clock())
        links, self._pending_push = self._pending_push, []
        if link is not None:
            links.append(link)
        for link in sorted(links, key=lambda lk: lk["seq"]):
            for peer in self._subscribers():
                try:
                    await self.transport.push(peer, link)
                except Exception:  # noqa: BLE001
                    log.debug("push to %s failed", peer, exc_info=True)

    def _subscribers(self) -> list[str]:
        rnd = self.round_at(self.clock())
        pollers = (self._draw(rnd, self.address) if rnd else [])
        return sorted({*self.roster(self.address), *pollers} - {self.address})

    async def _check_absent(self) -> None:
        """At the end of a round, report drawn pollers that never polled."""
        if self.worklog is None:
            return
        now = self.clock()
        rnd = self.round_at(now - self.params.early_slack_s - self.params.tick_s * 2)
        cur = self.round_at(now)
        if rnd is None or cur is None or cur.index == rnd.index or rnd.index in self._absent_done:
            return
        self._absent_done.add(rnd.index)
        drawn = self._draw(rnd, self.address)
        for slot, poller in enumerate(drawn):
            if poller not in self._polled_by.get(rnd.index, set()):
                await self._report("poller_absent", robot=self.address, round=rnd.index,
                                   poller=poller, slot=slot)

    # -- poller ------------------------------------------------------------------

    def duties(self, rnd: P.Round) -> list[tuple[str, int]]:
        """(robot, slot) pairs this node was drawn for in the round."""
        out = []
        for robot in self.robots():
            drawn = self._draw(rnd, robot)
            out += [(robot.lower(), s) for s, a in enumerate(drawn) if a == self.address]
        return out

    def _round(self, index: Any) -> P.Round | None:
        """The current round, or the one just ended while its slack lasts."""
        now = self.clock()
        for t in (now, now - self.params.early_slack_s):
            rnd = self.round_at(t)
            if rnd is not None and rnd.index == index:
                return rnd
        return None

    async def _poll_due(self) -> None:
        now = self.clock()
        rounds = {r.index: r for r in (self.round_at(now),
                                       self.round_at(now - self.params.early_slack_s)) if r}
        for rnd in rounds.values():
            await self._poll_round(rnd, now)

    async def _poll_round(self, rnd: P.Round, now: float) -> None:
        for robot, slot in self.duties(rnd):
            key = (rnd.index, robot, slot)
            if key in self._done:
                continue
            poll = P.make_poll(rnd, robot, slot, poller=self.address, private_key=self._key)
            at, _ = P.poll_timing(poll, rnd)
            if now >= at:
                self._done.add(key)
                await self.run_poll(rnd, poll)

    async def run_poll(self, rnd: P.Round, poll: dict) -> dict[str, list[str]]:
        """Send one poll to the robot and its witnesses at once. Returns the
        errors per answerer (empty list = countersigned)."""
        robot = poll["robot"]
        targets = [robot] + [w for w in self.roster(robot) if w != robot]

        async def ask(peer: str) -> tuple[str, dict | None, float, float]:
            held = self.heads.get(peer)
            since = held[0]["seq"] if held else -1
            sent = self.clock()
            try:
                ans = await self.transport.request(
                    peer, {"type": "poll", "poll": poll, "since": since},
                    timeout=self.params.deadline_s * 3)
            except Exception as exc:  # noqa: BLE001
                ans = {"error": f"no answer: {exc}"}
            return peer, ans, sent, self.clock()

        results = await asyncio.gather(*(ask(t) for t in targets))
        out: dict[str, list[str]] = {}
        for peer, ans, sent, received in results:
            role = "subject" if peer == robot else "witness"
            if not ans or "links" not in ans:
                err = (ans or {}).get("error", "no answer")
                out[peer] = [err]
                await self._report("no_answer", robot=peer, subject=robot, role=role,
                                   poll=P.poll_id(poll), error=err)
                continue
            held = self.heads.get(peer)
            errs = P.verify_answer(ans["links"], poll, rnd, answerer=peer,
                                   known_head=held[0] if held else None,
                                   sent_at=sent, received_at=received,
                                   deadline_s=self.params.deadline_s,
                                   required_packs=list(rnd.standards) or None)
            out[peer] = errs
            if errs:
                kind = ("late" if all(e.startswith("late") for e in errs)
                        else "wrong_packs" if errs == [P.WRONG_PACKS]
                        else "chain_break" if any(e.startswith("link ") for e in errs)
                        else "invalid")
                await self._report(kind, robot=peer, subject=robot, role=role,
                                   poll=P.poll_id(poll), errors=errs)
                continue
            link = ans["links"][-1]
            self.heads[peer] = (link, self.clock())
            receipt = P.make_receipt(poll, link, role=role, received_at=received,
                                     poller=self.address, private_key=self._key)
            self._my_receipts.setdefault(rnd.index, []).append(receipt)
            try:
                await self.transport.push(peer, receipt)
            except Exception:  # noqa: BLE001
                log.debug("receipt push to %s failed", peer, exc_info=True)
        return out

    async def _aggregate_finished(self) -> None:
        now = self.clock()
        cur = self.round_at(now)
        for idx in sorted(self._my_receipts):
            if idx in self._aggregated or (cur is not None and cur.index == idx):
                continue
            self._aggregated.add(idx)
            agg = P.make_aggregate(idx, self._my_receipts[idx], poller=self.address,
                                   private_key=self._key)
            await self.transport.gossip(agg)

    # -- reports and the loop ------------------------------------------------------

    async def _report(self, kind: str, **detail: Any) -> None:
        rep = P.make_report(kind, reporter=self.address, private_key=self._key,
                            now=self.clock(), **detail)
        self.reports_sent.append(rep)
        log.info("robot poll report: %s %s", kind, detail)
        try:
            await self.transport.gossip(rep)
        except Exception:  # noqa: BLE001
            log.debug("report gossip failed", exc_info=True)

    async def step(self) -> None:
        await self._commit_and_push()
        await self._poll_due()
        await self._watch()
        await self._check_absent()
        await self._aggregate_finished()

    async def _loop(self) -> None:
        while True:
            try:
                await self.step()
            except Exception:  # noqa: BLE001
                log.warning("poll node step failed", exc_info=True)
            await asyncio.sleep(self.params.tick_s)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None


# -- in-process transport ------------------------------------------------------

class LocalNetwork:
    """Every node in one process. ``down`` holds addresses that drop all
    traffic (a robot that went silent)."""

    def __init__(self) -> None:
        self.nodes: dict[str, PollNode] = {}
        self.down: set[str] = set()
        self.gossip_log: list[dict] = []

    def add(self, node: PollNode) -> None:
        self.nodes[node.address] = node

    def transport(self, address: str) -> "LocalTransport":
        return LocalTransport(self, address.lower())


class LocalTransport:
    def __init__(self, net: LocalNetwork, me: str) -> None:
        self.net, self.me = net, me

    def _peer(self, to: str) -> PollNode:
        to = to.lower()
        if self.me in self.net.down or to in self.net.down or to not in self.net.nodes:
            raise ConnectionError(f"{to} unreachable")
        return self.net.nodes[to]

    async def request(self, to: str, msg: dict, timeout: float) -> dict:
        peer = self._peer(to)
        return await asyncio.wait_for(peer.handle_request(json.loads(json.dumps(msg))), timeout)

    async def push(self, to: str, msg: dict) -> None:
        await self._peer(to).handle_push(json.loads(json.dumps(msg)))

    async def gossip(self, msg: dict) -> None:
        if self.me in self.net.down:
            return
        self.net.gossip_log.append(msg)
        for addr, node in self.net.nodes.items():
            if addr not in self.net.down:
                await node.handle_gossip(json.loads(json.dumps(msg)))

