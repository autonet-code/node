"""The poll rail inside a daemon (docs/robots.md, 2026-10-25, Phase A).

:class:`RailService` runs one :class:`~atn.robot.node.PollNode` per local
agent with a key: the robot agent (with its worklog) and every other agent,
any of which the lottery may draw as a poller. Its inputs all come from the
chain, so every daemon sees the same rounds, pollers and witnesses:

- **Rounds** (:class:`ChainRounds`). Round ``r`` starts at
  ``T0 + r * period_s`` (``T0`` = the first anchor's timestamp). It is
  pinned to the anchor in force at its start: REP and households from that
  anchor's voice snapshot, the seed from the epoch root two anchors before
  it, sanctions and territories from the DAO Registry at the same anchor.
- **Rosters**: the Registry's ``territory.<name>`` keys. Robots in one
  territory witness each other.
- **Peers**: agent address to libp2p peer from the on-chain registry.

Transport is the ``/autonet/robot/1.0.0`` protocol for requests and pushes,
and the ``autonet/robot/1`` gossip topic for aggregates and reports.
"""
from __future__ import annotations

import asyncio
import bisect
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import poll as P
from .node import PollNode, PollParams

log = logging.getLogger(__name__)

GOSSIP_TOPIC = "autonet/robot/1"


class ChainRounds:
    """Poll rounds derived from the chain. ``refresh`` does the blocking
    reads (run it off the event loop); ``round_at`` only reads the cache.

    ``anchors()`` returns ``[(timestamp, epoch_root_hex), ...]`` for every
    anchor; ``voice(i)`` returns the voice snapshot pinned to anchor ``i``
    (``read_voice_state(..., anchor_index=i)``)."""

    def __init__(self, *, period_s: float, anchors: Callable[[], list[tuple[int, str]]],
                 voice: Callable[[int], dict[str, Any]]) -> None:
        self.period_s = float(period_s)
        self._anchors_fn, self._voice_fn = anchors, voice
        self._lock = threading.Lock()
        self._anchors: list[tuple[int, str]] = []
        self._voices: dict[int, dict[str, Any]] = {}

    def refresh(self, now: float | None = None) -> None:
        anchors = self._anchors_fn()
        now = time.time() if now is None else now
        with self._lock:
            self._anchors = list(anchors)
        # Voice snapshots for the rounds around now (the current one, the
        # one just ended for its slack, the next for its subscribers).
        for t in (now - self.period_s, now, now + self.period_s):
            idx = self._anchor_for(t)
            if idx is not None and idx not in self._voices:
                v = self._voice_fn(idx)
                with self._lock:
                    self._voices[idx] = v
        with self._lock:
            keep = {self._anchor_for(t) for t in (now - self.period_s, now)}
            for idx in [i for i in self._voices if i not in keep and len(self._voices) > 4]:
                del self._voices[idx]

    def _round_number(self, t: float) -> int | None:
        if not self._anchors or t < self._anchors[0][0]:
            return None
        return int((t - self._anchors[0][0]) // self.period_s)

    def _anchor_for(self, t: float) -> int | None:
        """The anchor in force at the start of the round containing t, if
        two anchors precede it (the seed anchor must exist)."""
        r = self._round_number(t)
        if r is None:
            return None
        start = self._anchors[0][0] + r * self.period_s
        idx = bisect.bisect_right([a[0] for a in self._anchors], start) - 1
        return idx if idx >= 2 else None

    def round_at(self, t: float) -> P.Round | None:
        with self._lock:
            r = self._round_number(t)
            idx = self._anchor_for(t)
            voice = self._voices.get(idx) if idx is not None else None
            if r is None or voice is None:
                return None
            start = self._anchors[0][0] + r * self.period_s
            seed = self._anchors[idx - 2][1]
        owner_of = voice.get("owner_map") or {}
        members = tuple(
            P.Member(agent, str(owner_of.get(agent) or agent).lower(), int(rep))
            for agent, rep in sorted((voice.get("agent_rep") or {}).items()))
        return P.Round(index=r, start=float(start), seed=seed.removeprefix("0x").lower(),
                       period_s=self.period_s, members=members,
                       sanctioned=frozenset(voice.get("sanctioned_households") or ()))

    def territories(self, t: float) -> dict[str, list[str]]:
        with self._lock:
            idx = self._anchor_for(t)
            voice = self._voices.get(idx) if idx is not None else None
        return dict((voice or {}).get("territories") or {})


def chain_sources(cfg: Any) -> tuple[Callable, Callable]:
    """(anchors, voice) readers over the configured Substrate, RepToken and
    DAO Registry. Blocking; call off the event loop."""
    from web3 import Web3

    from nodes.common.voice_state import _VOICE_ABI, read_voice_state

    substrate = (getattr(cfg, "substrate_address", "")
                 or getattr(cfg, "rpb_contract_address", ""))
    rpc = getattr(cfg, "rpc_url", "")
    rep = getattr(cfg, "rep_token_address", "") or ""
    registry = getattr(cfg, "dao_registry_address", "") or ""
    w3 = Web3(Web3.HTTPProvider(rpc))
    contract = w3.eth.contract(address=Web3.to_checksum_address(substrate), abi=_VOICE_ABI)
    cache: list[tuple[int, str]] = []

    def anchors() -> list[tuple[int, str]]:
        count = int(contract.functions.anchorCount().call())
        for i in range(len(cache), count):    # anchors never change once written
            a = contract.functions.getAnchor(i).call()
            cache.append((int(a[8]), "0x" + bytes(a[1]).hex()))
        return list(cache)

    def voice(i: int) -> dict[str, Any]:
        return read_voice_state(substrate, rpc, rep_token_address=rep,
                                dao_registry_address=registry, web3=w3, anchor_index=i)
    return anchors, voice


class Libp2pTransport:
    """The rail over the daemon's libp2p host. Messages to an agent on this
    same daemon are delivered locally."""

    def __init__(self, rail: "RailService", host: Any, me: str) -> None:
        self.rail, self.host, self.me = rail, host, me

    def _peer(self, to: str) -> str:
        attribution = getattr(self.host, "_peer_attribution", None)
        rec = attribution.peer_for_agent(to) if attribution is not None else None
        if rec is None:
            raise ConnectionError(f"no peer id on chain for {to}")
        return rec.peer_id

    async def _send(self, to: str, envelope: dict, timeout: float) -> dict:
        local = self.rail.nodes.get(to.lower())
        if local is not None:
            return await self.rail.dispatch(envelope)
        import trio
        peer = self._peer(to)
        token = getattr(self.host, "_trio_token", None)
        if token is None:
            raise ConnectionError("libp2p host not running")

        def call() -> dict:
            async def _do():
                return await self.host.request_robot(peer, envelope, timeout=timeout)
            return trio.from_thread.run(_do, trio_token=token)
        return await asyncio.get_running_loop().run_in_executor(None, call)

    async def request(self, to: str, msg: dict, timeout: float) -> dict:
        return await self._send(to, {"to": to.lower(), "kind": "request", "msg": msg}, timeout)

    async def push(self, to: str, msg: dict) -> None:
        await self._send(to, {"to": to.lower(), "kind": "push", "msg": msg}, 10.0)

    async def gossip(self, msg: dict) -> None:
        await self.rail.on_gossip(msg)            # our own nodes see it too
        self.host.publish_topic_threadsafe(GOSSIP_TOPIC, json.dumps(msg).encode("utf-8"))


class RailService:
    def __init__(self, runtime: Any, config: Any) -> None:
        self._runtime = runtime
        self._config = config
        self.cfg = config.robot_poll
        self.params = PollParams(k=self.cfg.pollers, percentile=self.cfg.percentile,
                                 deadline_s=self.cfg.deadline_s,
                                 commit_gap_s=self.cfg.commit_gap_s)
        self.nodes: dict[str, PollNode] = {}
        self.rounds: ChainRounds | None = None
        self.state_dir = Path(config.data_dir) / "robot" / "rail"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []
        self.error = ""

    # -- inbound, from the host's trio thread ----------------------------------

    async def dispatch(self, envelope: dict) -> dict:
        node = self.nodes.get(str(envelope.get("to", "")).lower())
        if node is None:
            return {"error": "no such agent on this daemon"}
        if envelope.get("kind") == "request":
            return await node.handle_request(envelope.get("msg") or {})
        await node.handle_push(envelope.get("msg") or {})
        return {"ok": True}

    async def on_gossip(self, msg: dict) -> None:
        for node in self.nodes.values():
            await node.handle_gossip(msg)

    def _bridge(self, coro_fn: Callable[[], Any]) -> Any:
        """Run an asyncio coroutine from the trio thread, blocking only a
        worker thread of trio's."""
        import trio
        loop = self._loop

        def run():
            return asyncio.run_coroutine_threadsafe(coro_fn(), loop).result(30)
        return trio.to_thread.run_sync(run)

    def _install_host_handlers(self, host: Any) -> None:
        async def handle(envelope: dict) -> dict:
            return await self._bridge(lambda: self.dispatch(envelope))

        def on_topic(_topic: str, data: bytes) -> None:
            try:
                msg = json.loads(data.decode("utf-8"))
            except ValueError:
                return
            asyncio.run_coroutine_threadsafe(self.on_gossip(msg), self._loop)
        host.set_robot_handler(handle)
        host.subscribe_topic_threadsafe(GOSSIP_TOPIC, on_topic)

    # -- nodes -------------------------------------------------------------------

    def _roster(self, robot: str) -> list[str]:
        robot = robot.lower()
        for robots in (self.rounds.territories(time.time()) if self.rounds else {}).values():
            if robot in robots:
                return [r for r in robots if r != robot]
        return []

    def _robots(self) -> list[str]:
        terr = self.rounds.territories(time.time()) if self.rounds else {}
        return sorted({r for robots in terr.values() for r in robots})

    def _build_nodes(self, host: Any) -> None:
        rt = self._runtime
        robot = getattr(rt, "robot", None)
        robot_agent = self._config.robot.agent_id if robot is not None else ""
        for defn, _status in rt.registry.list_agents():
            agent_id = defn.id
            address = str(getattr(getattr(defn, "identity", None), "address", "") or "").lower()
            key = rt.registry.get_agent_key(agent_id)
            if not address or not key:
                continue
            is_robot = agent_id == robot_agent
            node = PollNode(
                address=address, private_key=key,
                transport=Libp2pTransport(self, host, address), params=self.params,
                round_at=self.rounds.round_at, roster=self._roster, robots=self._robots,
                state_dir=self.state_dir / address,
                worklog=robot.worklog if is_robot else None,
                packs=(lambda: list(robot.exported)) if is_robot else list,
                write_challenge=robot.write_challenge if is_robot else None)
            self.nodes[address] = node

    # -- lifecycle ---------------------------------------------------------------

    async def _refresh_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                await loop.run_in_executor(None, self.rounds.refresh)
                self.error = ""
            except Exception as exc:  # noqa: BLE001
                self.error = f"chain read failed: {exc}"
                log.warning("robot rail: %s", self.error)
            await asyncio.sleep(max(5.0, min(60.0, self.rounds.period_s / 10)))

    async def _prune_loop(self) -> None:
        robot = getattr(self._runtime, "robot", None)
        while True:
            await asyncio.sleep(60)
            if robot is not None and self.cfg.retention_rounds:
                retention = {s: n * self.cfg.period_s
                             for s, n in self.cfg.retention_rounds.items()}
                try:
                    robot.worklog.prune(retention)
                except Exception:  # noqa: BLE001
                    log.warning("worklog prune failed", exc_info=True)

    async def start(self) -> bool:
        """Start once the autonet network is up (it may come up after the
        runtime starts). Returns at once."""
        self._loop = asyncio.get_running_loop()
        self._tasks = [asyncio.create_task(self._boot())]
        return True

    async def _boot(self, timeout: float = 600.0) -> None:
        deadline = time.time() + timeout
        while True:
            autonet = getattr(self._runtime, "autonet", None)
            host = getattr(autonet, "_p2p_host", None) if autonet else None
            if host is not None and getattr(host, "_trio_token", None) is not None:
                break
            if time.time() > deadline:
                self.error = "the robot poll rail needs the autonet network (p2p and chain)"
                log.warning("robot rail not started: %s", self.error)
                return
            await asyncio.sleep(2.0)
        try:
            await self._start_on(autonet, host)
        except Exception as exc:  # noqa: BLE001
            self.error = f"start failed: {exc}"
            log.warning("robot rail: %s", self.error, exc_info=True)

    async def _start_on(self, autonet: Any, host: Any) -> None:
        anchors, voice = chain_sources(autonet.config)
        self.rounds = ChainRounds(period_s=self.cfg.period_s, anchors=anchors, voice=voice)
        self._build_nodes(host)
        self._install_host_handlers(host)
        self._tasks += [asyncio.create_task(self._refresh_loop()),
                        asyncio.create_task(self._prune_loop())]
        for node in self.nodes.values():
            node.start()
        log.info("robot rail: %d local agents", len(self.nodes))

    async def stop(self) -> None:
        for node in self.nodes.values():
            await node.stop()
        for t in self._tasks:
            t.cancel()
        self._tasks = []

    def status(self) -> dict[str, Any]:
        now = time.time()
        rnd = self.rounds.round_at(now) if self.rounds else None
        return {
            "round": rnd.index if rnd else None, "error": self.error,
            "robots": self._robots(),
            "agents": {
                a: {"duties": len(n.duties(rnd)) if rnd else 0,
                    "receipts": len(n.receipts), "reports_sent": len(n.reports_sent),
                    "watching": len(n.heads)}
                for a, n in self.nodes.items()},
        }
