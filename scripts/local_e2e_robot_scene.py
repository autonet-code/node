#!/usr/bin/env python3
"""Scene summaries and the automatic trigger, end to end (docs/robots.md,
Decision 2026-10-25, Phase B).

The same three-TurtleBot world as ``local_e2e_robot_poll.py`` (and its
Docker and worklog-sync helpers). Each robot's scene node writes a sample a
second; each daemon commits its samples in its signed links, and every robot
checks the others' claims against what its own lidar saw.

Stages
  1  sim up, every robot writing scene samples in world coordinates
  2  each robot's samples show the robots near it where they really are
  3  samples travel in the signed links; watchers learn each robot's position
  4  proximity witnesses: a poll of r1 goes to r2 (near), not r3 (far); r3,
     alone, has no witnesses
  5  honest driving: r1 drives out and back, nobody reports anything
  6  the lie: r1's daemon claims r1 stayed put while it drives away; r2, which
     could see the empty spot, reports a scene conflict; r3, out of range,
     says nothing; nothing else is reported

Run: python scripts/local_e2e_robot_scene.py   (needs Docker and the sim image)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import local_e2e_robot_poll as F  # noqa: E402  (shared sim, sync and fleet helpers)

from atn.robot import packs as PK  # noqa: E402
from atn.robot import poll as P  # noqa: E402
from atn.robot import worklog as W  # noqa: E402
from atn.robot.node import LocalNetwork, PollNode, PollParams  # noqa: E402
from atn.robot.rail import ChainRounds  # noqa: E402
from atn.robot.scene_source import SceneSource  # noqa: E402

RESULTS: list[tuple[int, str, str]] = []
SPAWN = {"r1": (-2.0, -0.5), "r2": (-2.0, 0.5), "r3": (-0.5, -2.0)}
RANGE_M = 1.8          # r1 and r2 are 1.0 m apart; r3 is 2.1 m from r1


def log(msg: str) -> None:
    print(f"[scene-e2e] {msg}", flush=True)


def check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(what)


class Liar:
    """A compromised daemon: it reports the robot frozen where the lie began."""

    def __init__(self, real: SceneSource, at: list[float]) -> None:
        self.real, self.at = real, at

    def between(self, t0: float, t1: float) -> list[dict]:
        return [{**s, "pose": list(self.at), "v": 0.0} for s in self.real.between(t0, t1)]

    def latest(self):
        return self.real.latest()


class Drive:
    def __init__(self, ws) -> None:
        self.ws = ws

    async def advertise(self, ns: str) -> None:
        await self.ws.send(json.dumps({"op": "advertise", "topic": f"/{ns}/cmd_vel",
                                       "type": "geometry_msgs/msg/TwistStamped"}))

    async def hold(self, ns: str, linear: float, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            await self.ws.send(json.dumps({
                "op": "publish", "topic": f"/{ns}/cmd_vel",
                "msg": {"header": {"frame_id": "base_link"},
                        "twist": {"linear": {"x": linear, "y": 0, "z": 0},
                                  "angular": {"x": 0, "y": 0, "z": 0}}}}))
            await asyncio.sleep(0.1)


async def amain() -> int:
    stage_no, stage_name = 0, ""

    def stage(n: int, name: str) -> None:
        nonlocal stage_no, stage_name
        stage_no, stage_name = n, name
        log(f"stage {n}: {name}")

    def passed(note: str = "") -> None:
        RESULTS.append((stage_no, stage_name, "PASS " + note))

    sync = F.WorklogSync()
    tasks: list[asyncio.Task] = []
    nodes: dict[str, PollNode] = {}
    try:
        stage(1, "sim up, scene samples flowing")
        F.start_sim()
        pack_src = (F.REPO / "robot" / "packs" / "proximity_stop.py").read_text("utf-8")
        manifest = {"name": "proximity_stop", "trust_class": "pinned", "runtime": "python3",
                    "code_digest": hashlib.sha256(pack_src.encode()).hexdigest()}
        for ns in SPAWN:
            digest = PK.write_bundle(F.SHARED / ns / "bundle", [(manifest, pack_src)])[0]
        tasks.append(asyncio.create_task(F.heartbeat_loop()))
        tasks.append(asyncio.create_task(F.sync_loop(sync)))
        sources = {ns: SceneSource(F.SHARED / ns / "bundle" / "scene.jsonl") for ns in SPAWN}
        check(await F.wait_for(lambda: all(len(s.between(0, 2e9)) >= 3
                                           for s in sources.values()), 240),
              "a robot wrote no scene samples")
        for ns, src in sources.items():
            pose = src.latest()["pose"]
            check(math.hypot(pose[0] - SPAWN[ns][0], pose[1] - SPAWN[ns][1]) < 0.15,
                  f"{ns} reports {pose[:2]}, spawned at {SPAWN[ns]}")
        passed(", ".join(f"{ns} at ({s.latest()['pose'][0]:.2f}, {s.latest()['pose'][1]:.2f})"
                         for ns, s in sources.items()))

        stage(2, "each robot sees its neighbours where they are")
        seen = []
        for ns, src in sources.items():
            me = src.latest()
            for other, osrc in sources.items():
                if other == ns:
                    continue
                there = osrc.latest()["pose"]
                d = math.hypot(there[0] - me["pose"][0], there[1] - me["pose"][1])
                if d > RANGE_M:
                    continue
                hit = min(math.hypot(o[0] - there[0], o[1] - there[1]) for o in me["objects"])
                check(hit < 0.4, f"{ns} sees no object at {other} ({hit:.2f} m off)")
                seen.append(f"{ns} sees {other} {hit:.2f} m off")
        check(seen, "no robot is near another")
        passed("; ".join(seen))

        stage(3, "samples in the signed links")
        accts = {ns: F._acct() for ns in SPAWN}
        addr = {ns: a for ns, (a, _k) in accts.items()}
        ns_of = {a: ns for ns, a in addr.items()}
        robots = sorted(addr.values())
        members = [F._acct() for _ in range(3)]
        keys = {a: k for a, k in list(accts.values()) + members}
        now = time.time()
        # Three anchors before the current round starts, so it has a seed.
        anchors = [(int(now - 700 + 10 * i), "0x" + hashlib.sha256(b"r%d" % i).hexdigest())
                   for i in range(3)]
        voice = {"agent_rep": {a: 100 * (i + 1) for i, (a, _k) in enumerate(members)}
                 | {r: 0 for r in robots}, "owner_map": {}, "sanctioned_households": [],
                 "robots": robots, "robot_standards": [digest]}
        rounds = ChainRounds(period_s=600.0, anchors=lambda: anchors, voice=lambda i: voice)
        rounds.refresh()
        params = PollParams(k=3, percentile=90, deadline_s=3.0, commit_gap_s=35.0,
                            witness_range_m=RANGE_M)
        net = LocalNetwork()

        def roster(robot: str) -> list[str]:
            return [r for r in robots if r != robot.lower()] if robot.lower() in robots else []
        for a in list(keys):
            ns = ns_of.get(a)
            node = PollNode(
                address=a, private_key=keys[a], transport=net.transport(a), params=params,
                round_at=rounds.round_at, roster=roster, robots=lambda: robots,
                state_dir=F.TMP / "scene_state" / a,
                worklog=W.Worklog(F.BAGS / ns, F.TMP / "scene_chain" / a, a, keys[a]) if ns else None,
                packs=(lambda: [digest]) if ns else list,
                write_challenge=(lambda c, ns=ns: PK.write_challenge(F.SHARED / ns / "bundle", c))
                if ns else None,
                scene=sources[ns] if ns else None)
            nodes[a] = node
            net.add(node)
            node.start()
        t_start = time.time()
        await asyncio.sleep(25)
        for ns, a in addr.items():
            links = nodes[a].worklog.links
            check(any(lk["scene"] for lk in links), f"{ns} committed no scene samples")
            check(W.verify_chain(links, robot=a) == [], f"{ns} chain broken")
            for other in robots:
                if other == a:
                    continue
                pos = nodes[other].positions.get(a)
                check(pos is not None and math.hypot(pos[0] - SPAWN[ns][0],
                                                     pos[1] - SPAWN[ns][1]) < 0.15,
                      f"{ns_of[other]} does not know where {ns} is")
        n_samples = sum(len(lk["scene"]) for a in robots for lk in nodes[a].worklog.links)
        passed(f"{n_samples} samples committed; every robot knows every position")

        stage(4, "proximity witnesses")
        poller = nodes[members[0][0]]
        check(await F.wait_for(lambda: all(r in poller.positions for r in robots), 20),
              "the poller does not know every position")
        w1 = sorted(ns_of[w] for w in poller._witnesses(addr["r1"]))
        w3 = sorted(ns_of[w] for w in poller._witnesses(addr["r3"]))
        check(w1 == ["r2"], f"r1's witnesses: {w1}")
        check(w3 == [], f"r3's witnesses: {w3}")
        passed("r1 -> r2; r3 -> none (2.1 m and 2.9 m away)")

        stage(5, "honest driving raises nothing")
        import websockets
        ws = await websockets.connect(F.ROSBRIDGE, max_size=None)
        drive = Drive(ws)
        await drive.advertise("r1")
        t5 = time.time()
        await drive.hold("r1", 0.25, 3.0)
        await drive.hold("r1", 0.0, 1.0)
        far = sources["r1"].latest()["pose"]
        check(far[0] > SPAWN["r1"][0] + 0.4, f"r1 did not move: {far[:2]}")
        await drive.hold("r1", -0.25, 3.0)
        await drive.hold("r1", 0.0, 1.0)
        await asyncio.sleep(25)                   # the drive's samples get committed
        reps = [g for g in net.gossip_log if g.get("type") == P.REPORT_TYPE
                and g.get("ts", 0) >= t_start]
        check(reps == [], f"reports during honest driving: {reps[:2]}")
        passed(f"r1 drove to x={far[0]:.2f} and back; 0 reports")

        stage(6, "the lie is caught by the robot that could see")
        r1 = addr["r1"]
        frozen = sources["r1"].latest()["pose"]
        nodes[r1].scene = Liar(sources["r1"], frozen)
        t6 = time.time()
        await drive.hold("r1", 0.25, 3.5)
        await drive.hold("r1", 0.0, 1.0)
        moved = sources["r1"].latest()["pose"]
        check(math.hypot(moved[0] - frozen[0], moved[1] - frozen[1]) > 0.5,
              f"r1 did not get away: {moved[:2]}")

        def conflicts() -> list[dict]:
            return [g for g in net.gossip_log if g.get("type") == P.REPORT_TYPE
                    and g.get("ts", 0) >= t6]
        check(await F.wait_for(lambda: any(g["kind"] == "scene_conflict" for g in conflicts()),
                               40), "nobody reported the lie")
        await asyncio.sleep(12)
        reps = conflicts()
        check({g["kind"] for g in reps} == {"scene_conflict"}, f"other reports: {reps[:2]}")
        check({g["robot"] for g in reps} == {r1}, "a report names an honest robot")
        reporters = {ns_of.get(g["reporter"], "member") for g in reps}
        check(reporters == {"r2"}, f"reported by {reporters}")
        unseen = sum(len(g["unseen"]) for g in reps)
        passed(f"r2 reported r1: {unseen} claimed positions it could see were empty; "
               f"r3 out of range")
        await ws.close()
        return 0
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((stage_no, stage_name, f"FAIL {type(exc).__name__}: {exc}"))
        try:
            print(F.docker("logs", "--tail", "30", F.CONTAINER, check_rc=False))
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        for node in nodes.values():
            await node.stop()
        for t in tasks:
            t.cancel()
        if os.environ.get("E2E_ROBOT_KEEP") != "1":
            F.stop_sim()
        print("\n=== robot scene E2E ===")
        for n, name, res in RESULTS:
            print(f"  {n:>2}  {name:<45} {res}")


if __name__ == "__main__":
    try:
        rc = asyncio.run(amain())
    finally:
        shutil.rmtree(F.TMP, ignore_errors=True)
    sys.exit(rc)
