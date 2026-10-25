#!/usr/bin/env python3
"""Robot MVP end to end (docs/robots.md, "MVP").

A simulated TurtleBot3 runs in Gazebo inside Docker (robot/sim/). The robot
daemon side runs in this process against a temp home. Every motion command
goes to ``/cmd_vel``; only the safety filter publishes the topic the wheels
listen on.

Stages
  1  sim up: rosbridge reachable, /scan and /odom flowing
  2  daemon exports the required pack, the filter loads it
  3  drive at an obstacle: the pack slows, then stops the robot
  4  reversing away is allowed
  5  a tampered pack is refused and all motion stops
  6  re-export restores motion
  7  daemon heartbeat goes stale: the robot stops
  8  sponsor unreachable: the robot stops
  9  challenge: the signed link verifies, the challenge lands in the next
     chunk; a tampered chunk and a wrong key fail
 10  the agent's ROS connector (ros-mcp) drives through the same filter
 11  the jurisdiction's standards (from the Registry, via the poll rail):
     a required pack this daemon lacks stops the robot; restoring the
     standards it can meet restores motion

Run: python scripts/local_e2e_robot.py   (needs Docker; builds the image once)
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# docker and uvx read the real home's config, so they get the original env.
_ORIG_ENV = dict(os.environ)
_TMP_HOME = Path(tempfile.mkdtemp(prefix="e2e_robot_home_"))
os.environ["KEYSTORE_DIR"] = str(_TMP_HOME / "keystore")
os.environ["USERPROFILE"] = str(_TMP_HOME)
os.environ["HOME"] = str(_TMP_HOME)
os.environ["ATN_DISABLE_TOOL_FLAG_KEEPER"] = "1"
os.environ["ATN_DISABLE_PROVIDER_AUTODETECT"] = "1"

import websockets  # noqa: E402

IMAGE = "autonet-robot-sim:dev"
CONTAINER = "autonet-robot-e2e"
ROSBRIDGE = "ws://127.0.0.1:9090"
SHARED = _TMP_HOME / "robot"            # mounted at /robot in the container
BUNDLE = SHARED / "bundle"
BAGS = SHARED / "bags"

RESULTS: list[tuple[int, str, str]] = []


def log(msg: str) -> None:
    print(f"[robot-e2e] {msg}", flush=True)


def check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(what)


# --- docker -------------------------------------------------------------------

def docker(*args: str, check_rc: bool = True, timeout: int = 1800) -> str:
    out = subprocess.run(["docker", *args], capture_output=True, text=True,
                         timeout=timeout, env=_ORIG_ENV)
    if check_rc and out.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])}: {out.stderr.strip()[-800:]}")
    return out.stdout


def start_sim() -> None:
    if not docker("images", "-q", IMAGE).strip():
        log("building the sim image (first run only)")
        docker("build", "-t", IMAGE, str(REPO / "robot" / "sim"))
    docker("rm", "-f", CONTAINER, check_rc=False)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    BAGS.mkdir(parents=True, exist_ok=True)
    docker("run", "-d", "--name", CONTAINER, "-p", "9090:9090",
           "-v", f"{REPO}:/autonet:ro", "-v", f"{SHARED}:/robot",
           IMAGE, "bash", "/autonet/robot/sim/run.sh")


def sync_worklog() -> None:
    """Copy the robot's worklog (container disk) to the daemon's bag dir."""
    shutil.rmtree(BAGS, ignore_errors=True)
    docker("cp", f"{CONTAINER}:/var/worklog", str(BAGS))


def stop_sim() -> None:
    docker("rm", "-f", CONTAINER, check_rc=False)


# --- rosbridge ----------------------------------------------------------------

class Ros:
    """Minimal rosbridge v2 client: subscribe to keep latest messages, publish."""

    def __init__(self) -> None:
        self.ws = None
        self.latest: dict[str, Any] = {}
        self.events: list[dict] = []
        self._reader = None

    async def connect(self, timeout: float = 240) -> None:
        deadline = time.time() + timeout
        while True:
            try:
                self.ws = await websockets.connect(ROSBRIDGE, max_size=None)
                break
            except Exception:  # noqa: BLE001 — docker accepts TCP before rosbridge is up
                if time.time() > deadline:
                    raise
                await asyncio.sleep(2)
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for raw in self.ws:
            m = json.loads(raw)
            if m.get("op") != "publish":
                continue
            topic = m["topic"]
            self.latest[topic] = (time.time(), m["msg"])
            if topic == "/safety/events":
                # Stamp with OUR clock: the container's clock can drift.
                self.events.append({**json.loads(m["msg"]["data"]), "_rx": time.time()})

    async def subscribe(self, topic: str, typ: str, throttle_ms: int = 0) -> None:
        await self.ws.send(json.dumps({"op": "subscribe", "topic": topic,
                                       "type": typ, "throttle_rate": throttle_ms}))

    async def advertise(self, topic: str, typ: str) -> None:
        await self.ws.send(json.dumps({"op": "advertise", "topic": topic, "type": typ}))

    async def drive(self, linear: float, angular: float) -> None:
        await self.ws.send(json.dumps({
            "op": "publish", "topic": "/cmd_vel",
            "msg": {"header": {"frame_id": "base_link"},
                    "twist": {"linear": {"x": linear, "y": 0, "z": 0},
                              "angular": {"x": 0, "y": 0, "z": angular}}}}))

    def odom_speed(self) -> float:
        _t, m = self.latest.get("/odom", (0, None))
        if m is None:
            return float("nan")
        tw = m["twist"]["twist"]
        return math.hypot(tw["linear"]["x"], tw["linear"]["y"])

    def front_min(self, half_angle_deg: float = 30.0) -> float:
        _t, m = self.latest.get("/scan", (0, None))
        if m is None:
            return float("nan")
        best = float("inf")
        for i, r in enumerate(m["ranges"]):
            a = m["angle_min"] + i * m["angle_increment"]
            a = math.atan2(math.sin(a), math.cos(a))
            if r is not None and abs(a) <= math.radians(half_angle_deg) \
                    and m["range_min"] <= r <= m["range_max"]:
                best = min(best, r)
        return best

    def yaw(self) -> float:
        q = self.latest["/odom"][1]["pose"]["pose"]["orientation"]
        return math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]),
                          1 - 2 * (q["y"] ** 2 + q["z"] ** 2))

    def target_bearing(self, lo: float = 1.3, hi: float = 2.5) -> float | None:
        """Bearing of an obstacle between lo and hi metres away, narrow-beam:
        every ray within 10 degrees of it must also be inside [lo, hi + 0.3]."""
        _t, m = self.latest["/scan"]
        rays = []
        for i, r in enumerate(m["ranges"]):
            a = m["angle_min"] + i * m["angle_increment"]
            rays.append((math.atan2(math.sin(a), math.cos(a)), r))
        for a, r in sorted(rays, key=lambda x: abs((x[1] or 99) - (lo + hi) / 2)):
            if r is None or not lo <= r <= hi:
                continue
            near = [rr for aa, rr in rays
                    if abs(math.atan2(math.sin(aa - a), math.cos(aa - a))) <= math.radians(10)]
            if all(rr is not None and lo - 0.2 <= rr <= hi + 0.3 for rr in near):
                return a
        return None

    def nearest_bearing(self) -> float:
        _t, m = self.latest["/scan"]
        best, bearing = float("inf"), 0.0
        for i, r in enumerate(m["ranges"]):
            if r is not None and m["range_min"] <= r <= m["range_max"] and r < best:
                best = r
                a = m["angle_min"] + i * m["angle_increment"]
                bearing = math.atan2(math.sin(a), math.cos(a))
        return bearing

    def events_since(self, t: float, kind: str | None = None) -> list[dict]:
        return [e for e in self.events if e["_rx"] >= t
                and (kind is None or e.get("kind") == kind)]

    async def hold(self, linear: float, angular: float, seconds: float) -> float:
        """Publish the command for ``seconds``; returns the peak odom speed
        seen during the last 1.5 s of it (after the robot had time to react)."""
        end = time.time() + seconds
        peak = 0.0
        while time.time() < end:
            await self.drive(linear, angular)
            await asyncio.sleep(0.1)
            if end - time.time() < 1.5:
                peak = max(peak, self.odom_speed())
        return peak


# --- daemon side --------------------------------------------------------------

def make_runtime():
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime
    data_dir, agents_dir = _TMP_HOME / "data", _TMP_HOME / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.robot.enabled = True
    config.robot.bundle_dir = str(BUNDLE)
    config.robot.bag_dir = str(BAGS)
    config.robot.heartbeat_interval_s = 0.5
    return Runtime(EventBus(), data_dir=data_dir, config=config), config


async def ws_call(rt, msg: dict) -> dict:
    from atn.ws_auth import ClientSession
    from atn.ws_server import WebSocketBridge
    bridge = WebSocketBridge(rt)
    session = ClientSession(local=True, authed=True, owner=True, scope_ids=None)
    return await bridge._handle_message({"msg_id": "e2e", **msg}, session)


async def start_robot_service(rt, config, *, probe: str) -> None:
    from atn.robot.service import RobotService
    if getattr(rt, "robot", None) is not None:
        await rt.robot.stop()
    if probe:
        os.environ["ATN_ROBOT_SPONSOR_PROBE"] = probe
    else:
        os.environ.pop("ATN_ROBOT_SPONSOR_PROBE", None)
    rt.robot = RobotService(rt, config)
    rt.robot.start()


async def wait_for(pred, timeout: float, step: float = 0.2) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


# --- stages -------------------------------------------------------------------

async def amain() -> int:
    from atn.agent_tools import _create_agent
    from atn.robot import worklog as W

    pack_src = (REPO / "robot" / "packs" / "proximity_stop.py").read_text("utf-8")
    ros = Ros()
    rt = config = None
    stage_no, stage_name = 0, ""

    def stage(n: int, name: str) -> None:
        nonlocal stage_no, stage_name
        stage_no, stage_name = n, name
        log(f"stage {n}: {name}")

    def passed(note: str = "") -> None:
        RESULTS.append((stage_no, stage_name, "PASS " + note))

    try:
        stage(1, "sim up")
        start_sim()
        await ros.connect()
        await ros.subscribe("/scan", "sensor_msgs/msg/LaserScan", 100)
        await ros.subscribe("/odom", "nav_msgs/msg/Odometry", 100)
        await ros.subscribe("/safety/events", "std_msgs/msg/String")
        await ros.advertise("/cmd_vel", "geometry_msgs/msg/TwistStamped")
        check(await wait_for(lambda: "/scan" in ros.latest and "/odom" in ros.latest,
                             240), "no /scan or /odom from the sim")
        passed(f"front {ros.front_min():.2f} m")

        stage(2, "daemon exports the pack, filter loads it")
        rt, config = make_runtime()
        reg = rt.tool_store.register(name="proximity_stop",
                                     description="Stop before obstacles, cap speed",
                                     input_schema={"type": "object"}, author="user",
                                     code=pack_src)
        rover = await _create_agent(rt, {"id": "rover", "provider": "external"})
        config.robot.required_packs = ["proximity_stop"]
        config.robot.agent_id = "rover"
        t0 = time.time()
        await start_robot_service(rt, config, probe="assume")
        check(rt.robot.exported == [reg["digest"]], f"export failed {rt.robot.export_errors}")
        check(await wait_for(lambda: any(e.get("packs") == [reg["digest"]]
                                         for e in ros.events_since(t0, "packs_loaded")), 15),
              "filter never loaded the pack")
        passed(reg["digest"][:12])

        stage(3, "drive at an obstacle: slowed, then stopped")
        bearing = ros.target_bearing()
        check(bearing is not None, "no obstacle 1.3 to 2.5 m away to drive at")
        goal = ros.yaw() + bearing
        turn_end = time.time() + 30
        while time.time() < turn_end:
            err = math.atan2(math.sin(goal - ros.yaw()), math.cos(goal - ros.yaw()))
            if abs(err) < 0.04:
                break
            await ros.hold(0.0, max(-0.5, min(0.5, 1.5 * err)), 0.1)
        await ros.hold(0, 0, 1.0)
        start_front = ros.front_min()
        t_drive = time.time()
        max_speed = 0.0
        end = time.time() + 40
        while time.time() < end:
            await ros.drive(0.8, 0.0)               # over the cap on purpose
            max_speed = max(max_speed, ros.odom_speed())
            await asyncio.sleep(0.1)
            if ros.events_since(t_drive, "verdict") and \
                    ros.events_since(t_drive, "verdict")[-1]["verdict"] == "stop":
                break
        await ros.hold(0.8, 0.0, 2.0)               # keep pushing while stopped
        stops = [e for e in ros.events_since(t_drive, "verdict") if e["verdict"] == "stop"]
        blind = [e for e in stops if "stale" in e["reason"]]
        stops = [e for e in stops if "obstacle" in e["reason"]]
        check(bool(stops), "the pack never stopped the robot for the obstacle")
        check(ros.odom_speed() < 0.03, f"still moving at {ros.odom_speed():.2f} m/s")
        check(max_speed <= 0.55, f"speed cap broken: {max_speed:.2f} m/s")
        check(ros.front_min() > 0.3, f"too close: {ros.front_min():.2f} m")
        passed(f"from {start_front:.2f} m to {ros.front_min():.2f} m, "
               f"peak {max_speed:.2f} m/s, {len(blind)} brief stale-sensor stops")

        stage(4, "reversing away is allowed")
        check(await ros.hold(-0.15, 0.0, 2.5) > 0.05, "reverse was blocked")
        await ros.hold(0, 0, 1.0)
        passed(f"front now {ros.front_min():.2f} m")

        stage(5, "tampered pack refused, motion stops")
        code_file = next((BUNDLE / "packs").glob("*.py"))
        t5 = time.time()
        code_file.write_text(pack_src.replace("0.45", "0.01"), "utf-8")
        check(await wait_for(lambda: bool(ros.events_since(t5, "packs_refused")), 10),
              "tampered pack was not refused")
        check(await ros.hold(-0.15, 0.0, 2.5) < 0.03, "robot moved with no valid pack")
        passed(ros.events_since(t5, "packs_refused")[0]["errors"][0][:60])

        stage(6, "re-export restores motion")
        t6 = time.time()
        check((await ws_call(rt, {"type": "robot_export_packs"}))["ok"], "re-export failed")
        check(await wait_for(lambda: bool(ros.events_since(t6, "packs_loaded")), 10),
              "pack not reloaded")
        check(await ros.hold(-0.15, 0.0, 2.5) > 0.05, "still blocked after restore")
        await ros.hold(0, 0, 1.0)
        passed()

        stage(7, "daemon heartbeat stale: stop")
        await rt.robot.stop()
        t7 = time.time()
        check(await wait_for(lambda: any("heartbeat stale" in e.get("reason", "")
                                         for e in ros.events_since(t7, "verdict")), 15),
              "no stale-heartbeat stop")
        check(await ros.hold(-0.15, 0.0, 2.5) < 0.03, "robot moved with a dead daemon")
        passed()

        stage(8, "sponsor unreachable: stop")
        await start_robot_service(rt, config, probe="")
        t8 = time.time()
        check(await wait_for(lambda: any("sponsor unreachable" in e.get("reason", "")
                                         for e in ros.events_since(t8 - 1, "verdict")), 15),
              "no sponsor-unreachable stop")
        check(await ros.hold(-0.15, 0.0, 2.5) < 0.03, "robot moved with no sponsor")
        passed()

        stage(9, "challenge")
        await start_robot_service(rt, config, probe="assume")
        await asyncio.sleep(12)                     # let a bag chunk close
        sync_worklog()
        nonce = uuid.uuid4().hex
        resp = await ws_call(rt, {"type": "robot_attest", "challenge": nonce})
        check(resp["ok"], f"attest failed: {resp}")
        link = resp["result"]
        check(W.verify_link(link, robot=rover["address"]) == [], "valid link rejected")
        check(link["challenge"] == nonce and link["packs"] == [reg["digest"]],
              "link does not carry the challenge and the active pack")
        files = rt.robot.worklog.produce(lf["name"] for lf in link["leaves"])
        check(files and W.verify_chunks(link, files) == [], "chunks do not match the link")
        first = link["leaves"][0]["name"]
        bad = dict(files)
        bad[first] = files[first][:-1] + b"\x00"
        check(W.verify_chunks(link, bad), "tampered chunk passed")
        check("signed by a different key" in W.verify_link(link, robot="0x" + "00" * 20),
              "wrong key passed")
        # The filter publishes the challenge; the next closed chunk records it.
        await asyncio.sleep(12)
        sync_worklog()
        later = [p.read_bytes() for p in BAGS.rglob("*.mcap") if p.name not in
                 {Path(lf["name"]).name for lf in link["leaves"]}]
        check(W.challenge_recorded(nonce, later), "challenge not recorded after the poll")
        passed(f"{len(link['leaves'])} chunks committed, challenge recorded")

        stage(10, "agent connector drives through the filter")
        res = await connector_drive(rt, ros)
        passed(res)

        stage(11, "Registry standards: a missing pack stops the robot")
        t11 = time.time()
        res = rt.robot.apply_standards([reg["digest"], "e" * 64])
        check(res is not None and not res["ok"], "an unavailable standard was accepted")
        check(await wait_for(lambda: bool(ros.events_since(t11, "packs_refused")), 10),
              "the filter did not refuse the emptied bundle")
        check(await ros.hold(-0.15, 0.0, 2.5) < 0.03, "robot moved without its standards")
        t11b = time.time()
        check(rt.robot.apply_standards([reg["digest"]])["ok"], "restoring standards failed")
        check(await wait_for(lambda: bool(ros.events_since(t11b, "packs_loaded")), 10),
              "pack not reloaded")
        check(await ros.hold(-0.15, 0.0, 2.5) > 0.05, "still blocked after the standards were met")
        await ros.hold(0, 0, 1.0)
        check(rt.robot.status()["required_from"] == "registry", "status does not name the source")
        passed("missing pack stops, met standards move")
        return 0
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((stage_no, stage_name, f"FAIL {type(exc).__name__}: {exc}"))
        try:
            print(docker("logs", "--tail", "60", CONTAINER, check_rc=False))
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        if rt is not None and getattr(rt, "robot", None) is not None:
            await rt.robot.stop()
        if rt is not None:
            try:
                await rt.connectors.stop_all()
            except Exception:  # noqa: BLE001
                pass
        if os.environ.get("E2E_ROBOT_KEEP") != "1":
            stop_sim()
        print("\n=== robot E2E ===")
        for n, name, res in RESULTS:
            print(f"  {n:>2}  {name:<45} {res}")


async def connector_drive(rt, ros: Ros) -> str:
    """The robot agent's ROS skills come from ros-mcp (an MCP connector). A
    forward command sent through it toward the obstacle is still stopped."""
    from atn.connectors_manager import ConnectorSpec
    spec = ConnectorSpec(mode="uvx", package="ros-mcp", name="ROS",
                         env={"ROSBRIDGE_IP": "127.0.0.1", "ROSBRIDGE_PORT": "9090"})
    rt.connectors.register("ros", spec)
    await rt.connectors.ensure_started(["ros"])
    names = [t.name.removeprefix("mcp_ros_") for t in rt.connectors.get_tools("ros")]
    publish = "publish_for_durations"
    check(publish in names, f"no {publish} in ros-mcp: {names[:30]}")
    msg = {"header": {"frame_id": "base_link"},
           "twist": {"linear": {"x": 0.4, "y": 0, "z": 0},
                     "angular": {"x": 0, "y": 0, "z": 0}}}
    # Forward at 0.4 m/s for 6 s, straight at the obstacle stage 3 stopped at.
    t = time.time()
    closest = [ros.front_min()]

    async def watch():
        while True:
            closest.append(ros.front_min())
            await asyncio.sleep(0.1)
    watcher = asyncio.create_task(watch())
    first = await rt.connectors.call_tool("ros", publish, {
        "topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/TwistStamped",
        "messages": [msg], "durations": [6.0], "rate_hz": 10})
    await asyncio.sleep(1.0)
    watcher.cancel()
    verdicts = [v["verdict"] for v in ros.events_since(t, "verdict")]
    check("published_count" in str(first), f"ros-mcp did not publish: {str(first)[:200]}")
    check(any(v != "pass" for v in verdicts), f"the filter never acted: {verdicts}")
    check(min(closest) >= 0.40, f"got within {min(closest):.2f} m")
    check(ros.odom_speed() < 0.03, "robot kept moving after the agent stopped")
    verdicts = sorted(set(verdicts))
    return (f"{len(names)} ros tools; forward 0.4 m/s for 6 s, closest "
            f"{min(closest):.2f} m, verdicts {verdicts}")



if __name__ == "__main__":
    try:
        rc = asyncio.run(amain())
    finally:
        shutil.rmtree(_TMP_HOME, ignore_errors=True)
    sys.exit(rc)
