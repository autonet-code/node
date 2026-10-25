#!/usr/bin/env python3
"""Poll rail end to end (docs/robots.md, Decision 2026-10-25, Phase A).

Three simulated TurtleBot3s share one Gazebo world in Docker
(``robot/sim/run_fleet.sh``), each with its own safety filter and a
two-stream MCAP worklog (pose, scan). Their daemons and five member
households run in this process as poll-rail nodes over the in-process
transport; rounds come from :class:`atn.robot.rail.ChainRounds` over a
fixed anchor history. The three robots form one territory, so each
witnesses the other two.

Stages
  1  sim up: three robots, both worklog streams growing
  2  each robot's filter loads its pack
  3  continuous commitment: every chain grows and verifies, witnesses and
     drawn pollers hold each robot's head
  4  a full round: every robot polled by each of its drawn pollers, every
     poll answered by the robot and both witnesses, receipts archived
  5  each challenge is recorded in the chunks closed after the poll, in the
     subject's worklog AND in both witnesses'
  6  each poller gossips one aggregate; every receipt is in one
  7  a dispute opens a countersigned window: chunks match, a tampered chunk
     is caught
  8  thinning: the scan stream is pruned; the pose leaves still verify
  9  a robot goes silent and a poller stays away for a round: missing
     commitment, unanswered polls and the absent poller are all reported,
     and nothing else is

Run: python scripts/local_e2e_robot_poll.py   (needs Docker and the sim image)
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eth_account import Account  # noqa: E402

from atn.robot import crypto as C  # noqa: E402
from atn.robot import packs as PK  # noqa: E402
from atn.robot import poll as P  # noqa: E402
from atn.robot import worklog as W  # noqa: E402
from atn.robot.node import LocalNetwork, PollNode, PollParams  # noqa: E402
from atn.robot.rail import ChainRounds  # noqa: E402

IMAGE = "autonet-robot-sim:dev"
CONTAINER = "autonet-fleet-e2e"
ROSBRIDGE = "ws://127.0.0.1:9090"
PERIOD_S = 45.0
N_ROBOTS = 3

TMP = Path(tempfile.mkdtemp(prefix="e2e_robot_poll_"))
SHARED = TMP / "robot"          # mounted at /robot (bundles)
BAGS = TMP / "bags"             # host copies of the worklogs
RESULTS: list[tuple[int, str, str]] = []


def log(msg: str) -> None:
    print(f"[poll-e2e] {msg}", flush=True)


def check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(what)


# --- docker -------------------------------------------------------------------

def docker(*args: str, check_rc: bool = True, timeout: int = 1800,
           binary: bool = False):
    out = subprocess.run(["docker", *args], capture_output=True, timeout=timeout,
                         text=not binary)
    if check_rc and out.returncode != 0:
        err = out.stderr if not binary else out.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"docker {' '.join(args[:3])}: {err.strip()[-800:]}")
    return out.stdout


def start_sim() -> None:
    if not docker("images", "-q", IMAGE).strip():
        log("building the sim image (first run only)")
        docker("build", "-t", IMAGE, str(REPO / "robot" / "sim"))
    docker("rm", "-f", CONTAINER, check_rc=False)
    SHARED.mkdir(parents=True, exist_ok=True)
    docker("run", "-d", "--name", CONTAINER, "-p", "9090:9090",
           "-e", f"ROBOTS={N_ROBOTS}", "-v", f"{REPO}:/autonet:ro",
           "-v", f"{SHARED}:/robot", IMAGE, "bash", "/autonet/robot/sim/run_fleet.sh")


def stop_sim() -> None:
    docker("rm", "-f", CONTAINER, check_rc=False)


class WorklogSync:
    """Mirrors the container's worklogs into BAGS (a Windows bind mount
    cannot carry bag rollover). Only closed chunks are copied; the chunk
    being written appears as an empty placeholder with its mtime, so the
    host worklog sees the same 'newest = still open' picture."""

    def __init__(self) -> None:
        self.copied: dict[str, tuple[float, int]] = {}
        self.paused: set[str] = set()     # robots whose sync is cut (silent)

    def sync(self) -> int:
        out = docker("exec", CONTAINER, "find", "/var/worklog", "-name", "*.mcap",
                     "-printf", "%P %T@ %s\n")
        by_stream: dict[tuple[str, str], list[tuple[str, float, int]]] = {}
        for line in out.splitlines():
            rel, mt, size = line.rsplit(" ", 2)
            robot, stream = rel.split("/")[:2]
            by_stream.setdefault((robot, stream), []).append((rel, float(mt), int(size)))
        want, opens = [], []
        for (robot, _stream), files in by_stream.items():
            if robot in self.paused:
                continue
            files.sort(key=lambda f: (f[1], f[0]))
            for rel, mt, size in files[:-1]:
                if self.copied.get(rel) != (mt, size):
                    want.append((rel, mt, size))
            opens.append(files[-1])
        if want:
            data = docker("exec", CONTAINER, "tar", "-cf", "-", "-C", "/var/worklog",
                          *[w[0] for w in want], binary=True)
            with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                tf.extractall(BAGS, filter="data")
            for rel, mt, size in want:
                os.utime(BAGS / rel, (mt, mt))
                self.copied[rel] = (mt, size)
        for rel, mt, _size in opens:
            p = BAGS / rel
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"")
                os.utime(p, (mt, mt))
        return len(want)


# --- rosbridge ----------------------------------------------------------------

class Ros:
    def __init__(self) -> None:
        self.events: dict[str, list[dict]] = {}
        self.ws = None

    async def connect(self, timeout: float = 240) -> None:
        import websockets
        deadline = time.time() + timeout
        while True:
            try:
                self.ws = await websockets.connect(ROSBRIDGE, max_size=None)
                break
            except Exception:  # noqa: BLE001 (docker accepts TCP before rosbridge is up)
                if time.time() > deadline:
                    raise
                await asyncio.sleep(2)
        asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for raw in self.ws:
            m = json.loads(raw)
            if m.get("op") == "publish":
                self.events.setdefault(m["topic"], []).append(json.loads(m["msg"]["data"]))

    async def subscribe(self, topic: str) -> None:
        await self.ws.send(json.dumps({"op": "subscribe", "topic": topic,
                                       "type": "std_msgs/msg/String"}))


# --- the fleet ----------------------------------------------------------------

def _acct():
    a = Account.create()
    return a.address.lower(), a.key.hex()


class Fleet:
    def __init__(self) -> None:
        self.net = LocalNetwork()
        self.keys: dict[str, str] = {}
        self.robots: list[str] = []
        self.ns: dict[str, str] = {}
        members = []
        for i in range(5):                       # member households, REP rising
            addr, key = _acct()
            self.keys[addr] = key
            members.append((addr, (i + 1) * 100))
        for i in range(N_ROBOTS):
            addr, key = _acct()
            self.keys[addr] = key
            self.robots.append(addr)
            self.ns[addr] = f"r{i + 1}"
        now = time.time()
        # A fixed anchor history: three anchors before now, so rounds have a seed.
        self.t0 = now - 3 * PERIOD_S - 5
        self.anchors = [(int(self.t0 + 10 * i), "0x" + hashlib.sha256(b"root%d" % i).hexdigest())
                        for i in range(3)]
        voice = {"agent_rep": {a: r for a, r in members} | {r: 0 for r in self.robots},
                 "owner_map": {}, "sanctioned_households": [],
                 "territories": {"yard": sorted(self.robots)}}
        self.rounds = ChainRounds(period_s=PERIOD_S, anchors=lambda: self.anchors,
                                  voice=lambda i: voice)
        self.rounds.refresh()
        self.params = PollParams(k=3, percentile=90, deadline_s=3.0, commit_gap_s=35.0)
        self.nodes: dict[str, PollNode] = {}
        for addr in [a for a, _ in members] + self.robots:
            is_robot = addr in self.robots
            ns = self.ns.get(addr, "")
            node = PollNode(
                address=addr, private_key=self.keys[addr],
                transport=self.net.transport(addr), params=self.params,
                round_at=self.rounds.round_at, roster=self.roster,
                robots=lambda: list(self.robots), state_dir=TMP / "state" / addr,
                worklog=W.Worklog(BAGS / ns, TMP / "chain" / addr, addr, self.keys[addr])
                if is_robot else None,
                packs=(lambda: [self.pack_digest]) if is_robot else list,
                write_challenge=(lambda c, ns=ns: PK.write_challenge(SHARED / ns / "bundle", c))
                if is_robot else None)
            self.nodes[addr] = node
            self.net.add(node)
        self.members = [a for a, _ in members]
        self.pack_digest = ""

    def roster(self, robot: str) -> list[str]:
        robot = robot.lower()
        return [r for r in self.robots if r != robot] if robot in self.robots else []

    def round_start(self, index: int) -> float:
        return self.anchors[0][0] + index * PERIOD_S

    def reports(self, since: float = 0.0) -> list[dict]:
        return [g for g in self.net.gossip_log
                if g.get("type") == P.REPORT_TYPE and g.get("ts", 0) >= since]


async def heartbeat_loop() -> None:
    while True:
        for i in range(N_ROBOTS):
            PK.write_heartbeat(SHARED / f"r{i + 1}" / "bundle", sponsor_ok=True)
        await asyncio.sleep(0.5)


async def sync_loop(sync: WorklogSync) -> None:
    while True:
        try:
            await asyncio.get_running_loop().run_in_executor(None, sync.sync)
        except Exception as exc:  # noqa: BLE001
            log(f"sync failed: {exc}")
        await asyncio.sleep(2.0)


async def wait_until(t: float) -> None:
    while time.time() < t:
        await asyncio.sleep(min(1.0, t - time.time()))


async def wait_for(pred, timeout: float, step: float = 1.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


# --- stages -------------------------------------------------------------------

async def amain() -> int:
    stage_no, stage_name = 0, ""

    def stage(n: int, name: str) -> None:
        nonlocal stage_no, stage_name
        stage_no, stage_name = n, name
        log(f"stage {n}: {name}")

    def passed(note: str = "") -> None:
        RESULTS.append((stage_no, stage_name, "PASS " + note))

    fleet = Fleet()
    sync = WorklogSync()
    tasks: list[asyncio.Task] = []
    try:
        stage(1, "sim up")
        start_sim()
        pack_src = (REPO / "robot" / "packs" / "proximity_stop.py").read_text("utf-8")
        manifest = {"name": "proximity_stop", "trust_class": "pinned", "runtime": "python3",
                    "code_digest": hashlib.sha256(pack_src.encode()).hexdigest()}
        for i in range(N_ROBOTS):
            digests = PK.write_bundle(SHARED / f"r{i + 1}" / "bundle", [(manifest, pack_src)])
        fleet.pack_digest = digests[0]
        tasks.append(asyncio.create_task(heartbeat_loop()))
        ros = Ros()
        await ros.connect()
        for i in range(N_ROBOTS):
            await ros.subscribe(f"/r{i + 1}/safety/events")
        tasks.append(asyncio.create_task(sync_loop(sync)))

        def grown() -> bool:
            for ns in fleet.ns.values():
                for stream in ("pose", "scan"):
                    if len(list((BAGS / ns / stream).rglob("*.mcap"))) < 3:
                        return False
            return True
        check(await wait_for(grown, 240), "worklogs did not grow on every robot")
        passed(f"{N_ROBOTS} robots, pose and scan streams")

        stage(2, "every filter loads its pack")

        def loaded(ns: str) -> bool:
            return any(e.get("kind") in ("packs_loaded", "status") and
                       e.get("packs") == [fleet.pack_digest]
                       for e in ros.events.get(f"/{ns}/safety/events", []))
        check(await wait_for(lambda: all(loaded(ns) for ns in fleet.ns.values()), 30),
              "a filter did not load the pack")
        passed(fleet.pack_digest[:12])

        stage(3, "continuous commitment")
        for node in fleet.nodes.values():
            node.start()
        await asyncio.sleep(25)
        for r in fleet.robots:
            wl = fleet.nodes[r].worklog
            check(len(wl.links) >= 2, f"{fleet.ns[r]} committed {len(wl.links)} links")
            check(W.verify_chain(wl.links, robot=r) == [], f"{fleet.ns[r]} chain broken")
            for w in fleet.roster(r):
                held = fleet.nodes[w].heads.get(r)
                check(held is not None and held[0]["seq"] >= wl.links[-2]["seq"],
                      f"witness {fleet.ns[w]} lags {fleet.ns[r]}")
        passed(", ".join(f"{fleet.ns[r]} {len(fleet.nodes[r].worklog.links)} links"
                         for r in fleet.robots))

        stage(4, "a full round")
        cur = fleet.rounds.round_at(time.time())
        R = cur.index + 1
        rnd_R = None
        log(f"waiting for round {R} to finish (~{fleet.round_start(R + 1) - time.time():.0f}s)")
        await wait_until(fleet.round_start(R) + 1)
        rnd_R = fleet.rounds.round_at(time.time())
        await wait_until(fleet.round_start(R + 1) + fleet.params.early_slack_s + 4)
        reports_R = [g for g in fleet.reports() if g.get("round", R) == R]
        check(reports_R == [], f"reports in a clean round: {reports_R[:3]}")
        for r in fleet.robots:
            drawn = P.draw_pollers(rnd_R, r, k=3, percentile=90)
            got = [x for x in fleet.nodes[r].receipts if x["round"] == R and x["subject"] == r]
            check(sorted(x["poller"] for x in got) == sorted(drawn),
                  f"{fleet.ns[r]}: receipts from {len(got)} of {len(drawn)} pollers")
            for w in fleet.roster(r):
                wg = [x for x in fleet.nodes[w].receipts
                      if x["round"] == R and x["subject"] == r and x["role"] == "witness"]
                check(len(wg) == len(drawn), f"witness {fleet.ns[w]} answered "
                      f"{len(wg)} of {len(drawn)} polls of {fleet.ns[r]}")
        passed(f"round {R}: 3 robots x 3 pollers, each answered by 3 worklogs")

        stage(5, "challenges recorded after the poll")
        await asyncio.sleep(22)          # two more chunks close and sync
        n_checked = 0
        for r in fleet.robots:
            for rc in [x for x in fleet.nodes[r].receipts if x["round"] == R]:
                wl = fleet.nodes[r].worklog
                link = next(lk for lk in wl.links if lk["seq"] == rc["seq"])
                later = [lf["name"] for lk in wl.links_since(link["seq"])
                         for lf in lk["leaves"] if lf["stream"] == "pose"]
                check(later, f"{fleet.ns[r]}: no pose chunk committed after the poll")
                files = wl.produce(later)
                check(W.challenge_recorded(link["challenge"], files.values()),
                      f"{fleet.ns[r]} ({rc['role']}): challenge not in later chunks")
                n_checked += 1
        passed(f"{n_checked} answers, challenge found in every later worklog")

        stage(6, "aggregates")
        aggs = [g for g in fleet.net.gossip_log if g.get("type") == P.AGGREGATE_TYPE
                and g["round"] == R]
        all_receipts = [x for r in fleet.robots for x in fleet.nodes[r].receipts if x["round"] == R]
        check(all(any(P.receipt_in_aggregate(x, a) for a in aggs) for x in all_receipts),
              "a receipt is in no aggregate")
        passed(f"{len(aggs)} aggregates cover {len(all_receipts)} receipts")

        stage(7, "a dispute opens a countersigned window")
        r1 = fleet.robots[0]
        wl = fleet.nodes[r1].worklog
        rc = next(x for x in fleet.nodes[r1].receipts if x["round"] == R and x["role"] == "subject")
        link = next(lk for lk in wl.links if lk["seq"] == rc["seq"])
        window = [lk for lk in wl.links if lk["seq"] <= link["seq"]][-3:]
        check(P.verify_receipt(rc, link) == [], "receipt does not match the link")
        names = [lf["name"] for lk in window for lf in lk["leaves"]]
        files = wl.produce(names)
        check(files and all(W.verify_chunks(lk, files) == [] for lk in window),
              "produced chunks do not match")
        victim = BAGS / fleet.ns[r1] / names[0]
        original = victim.read_bytes()
        victim.write_bytes(original[:-8] + b"\x00" * 8)
        bad = wl.produce(names)
        caught = any(W.verify_chunks(lk, bad) for lk in window)
        victim.write_bytes(original)
        check(caught, "a tampered chunk passed")
        passed(f"{len(files)} chunks over {len(window)} links, tamper caught")

        stage(8, "thinning the scan stream")
        removed = wl.prune({"scan": 0.0})
        check(removed and all("/scan/" in f"/{n}" for n in removed), "scan not pruned")
        kept = wl.produce(lf["name"] for lk in wl.links for lf in lk["leaves"])
        check(kept and not any(n.startswith("scan/") for n in kept), "scan chunks kept")
        check(all(W.verify_chunks(lk, kept) == [] for lk in wl.links), "pose chunks broken")
        check(W.verify_chain(wl.links, robot=r1) == [], "chain broken by thinning")
        passed(f"{len(removed)} scan chunks removed, {len(kept)} pose chunks verify")

        stage(9, "a silent robot and an absent poller")
        # Cut both in time for the whole of round R + 2.
        await wait_until(fleet.round_start(R + 2) - 3)
        silent = fleet.robots[2]
        rnd_next = fleet.rounds.round_at(fleet.round_start(R + 2) + 1)
        lazy = P.draw_pollers(rnd_next, fleet.robots[0], k=3, percentile=90)[0]
        fleet.net.down |= {silent, lazy}
        sync.paused.add(fleet.ns[silent])
        t_cut = time.time()
        await wait_until(fleet.round_start(R + 3) + fleet.params.early_slack_s + 4)
        reps = fleet.reports(since=t_cut)
        kinds = {(g["kind"], g.get("robot"), g.get("poller")) for g in reps}
        check(any(k == "missing_commit" and rb == silent for k, rb, _ in kinds),
              "no missing-commit report for the silent robot")
        check(any(k == "no_answer" and rb == silent for k, rb, _ in kinds),
              "no unanswered-poll report for the silent robot")
        for r in fleet.robots[:2]:
            check(("poller_absent", r, lazy) in kinds,
                  f"{fleet.ns[r]} did not report the absent poller")
        unexpected = [g for g in reps if not (
            (g["kind"] in ("missing_commit", "no_answer") and g.get("robot") == silent)
            or (g["kind"] == "poller_absent" and g.get("poller") == lazy))]
        check(unexpected == [], f"unexpected reports: {unexpected[:3]}")
        check(all(C.signer(g) == g["reporter"] for g in reps), "unsigned report")
        passed(f"{len(reps)} reports, all about {fleet.ns[silent]} or the absent poller")
        return 0
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((stage_no, stage_name, f"FAIL {type(exc).__name__}: {exc}"))
        try:
            print(docker("logs", "--tail", "40", CONTAINER, check_rc=False))
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        for node in fleet.nodes.values():
            await node.stop()
        for t in tasks:
            t.cancel()
        if os.environ.get("E2E_ROBOT_KEEP") != "1":
            stop_sim()
        print("\n=== robot poll E2E ===")
        for n, name, res in RESULTS:
            print(f"  {n:>2}  {name:<45} {res}")


if __name__ == "__main__":
    try:
        rc = asyncio.run(amain())
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(rc)
