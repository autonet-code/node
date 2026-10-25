"""ROS 2 safety filter node (docs/robots.md, "The seam: ROS 2").

Sits at the actuation boundary. Every command source (Nav2, teleop, the agent
through its MCP connector) publishes on ``in_topic``; only this node publishes
``out_topic``, the topic the motors listen on. At a fixed rate it takes the
latest proposed command (zero once it is older than ``cmd_timeout_s``), runs
the required safety packs from the daemon's bundle, and publishes the result.
Every non-pass verdict and a once-a-second status go out on ``safety/events``
as JSON, so they land in the worklog. A poll's challenge, handed over by the
daemon in the bundle, goes out on ``autonet/challenge`` for the same reason:
the next chunk then proves the robot recorded it after the poll.

The pack runtime is ``atn/robot/packs.py``, loaded by path: it is stdlib only
and this node runs under the robot's ROS Python, not the daemon's.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def _load_runtime(path: str):
    spec = importlib.util.spec_from_file_location("atn_robot_packs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atn_robot_packs"] = mod
    spec.loader.exec_module(mod)
    return mod


class SafetyFilter(Node):
    def __init__(self) -> None:
        super().__init__("autonet_safety_filter")
        p = self.declare_parameter
        self.bundle_dir = p("bundle_dir", "/robot/bundle").value
        runtime_path = p("runtime_path", "/autonet/atn/robot/packs.py").value
        in_topic = p("in_topic", "cmd_vel").value
        out_topic = p("out_topic", "cmd_vel_safe").value
        self.stamped = p("stamped", True).value
        self.out_stamped = p("out_stamped", True).value
        self.rate_hz = float(p("rate_hz", 20.0).value)
        self.cmd_timeout_s = float(p("cmd_timeout_s", 0.5).value)
        # How old the newest scan may be. Set it from the lidar's rate: a few
        # missed frames, never more than the robot can travel safely blind.
        self.obs_max_age_s = float(p("obs_max_age_s", 0.5).value)
        self.front_half_angle = math.radians(float(p("front_half_angle_deg", 30.0).value))

        self.packs_rt = _load_runtime(runtime_path)
        self.packs = []
        self.load_errors = ["not loaded yet"]
        self._required_mtime = None
        self.heartbeat = {}
        self.last_cmd = None
        self.last_cmd_t = 0.0
        self.scan_obs = None
        self.last_verdict = None

        msg_type = TwistStamped if self.stamped else Twist
        self.create_subscription(msg_type, in_topic, self._on_cmd, 10)
        self.create_subscription(LaserScan, "scan", self._on_scan,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(
            TwistStamped if self.out_stamped else Twist, out_topic, 10)
        self.events = self.create_publisher(String, "safety/events", 10)
        self.challenges = self.create_publisher(String, "autonet/challenge", 10)
        self._challenge_sent: set[str] = set()
        self._challenge_out: list[list] = []
        # Announce the topic at once, so recorders subscribe before a poll.
        self.challenges.publish(String(data="{}"))
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.create_timer(0.2, self._challenge)
        self.create_timer(1.0, self._refresh)
        self._refresh()
        self.get_logger().info(
            f"safety filter {in_topic} -> {out_topic}, bundle {self.bundle_dir}")

    # -- inputs -------------------------------------------------------------

    def _on_cmd(self, msg) -> None:
        tw = msg.twist if self.stamped else msg
        self.last_cmd = {"linear": float(tw.linear.x), "angular": float(tw.angular.z)}
        self.last_cmd_t = time.time()

    def _on_scan(self, msg: LaserScan) -> None:
        front = math.inf
        for i, r in enumerate(msg.ranges):
            a = msg.angle_min + i * msg.angle_increment
            a = math.atan2(math.sin(a), math.cos(a))
            if abs(a) <= self.front_half_angle and math.isfinite(r) \
                    and msg.range_min <= r <= msg.range_max:
                front = min(front, r)
        self.scan_obs = {"stamp": time.time(),
                         "scan": {"front_min": front if math.isfinite(front)
                                  else float(msg.range_max)}}

    def _refresh(self) -> None:
        self.heartbeat = self.packs_rt.read_heartbeat(self.bundle_dir)
        # Reload whenever anything in the bundle changes, so a pack edited in
        # place is re-verified (and refused) within a second.
        sig = []
        for root, _dirs, files in os.walk(self.bundle_dir):
            for f in sorted(files):
                if f.startswith(("heartbeat", "challenge")):
                    continue
                try:
                    st = os.stat(os.path.join(root, f))
                    sig.append((f, st.st_mtime_ns, st.st_size))
                except OSError:
                    pass
        sig = tuple(sorted(sig))
        if sig != self._required_mtime:
            self._required_mtime = sig
            res = self.packs_rt.load_bundle(self.bundle_dir)
            self.packs = res.packs if res.ok else []
            self.load_errors = res.errors
            self._event({"kind": "packs_loaded" if res.ok else "packs_refused",
                         "packs": [pk.digest for pk in res.packs],
                         "errors": res.errors})
        self._event({"kind": "status",
                     "packs": [pk.digest for pk in self.packs],
                     "errors": self.load_errors,
                     "heartbeat": self.heartbeat,
                     "verdict": (self.last_verdict or {}).get("verdict")})

    def _challenge(self) -> None:
        # Publish every challenge not yet sent, each for a second: a recorder
        # that only just discovered the topic may miss the first message.
        queue = self.packs_rt.read_challenges(self.bundle_dir)
        self._challenge_sent &= {c["challenge"] for c in queue}
        for c in queue:
            if c["challenge"] not in self._challenge_sent:
                self._challenge_sent.add(c["challenge"])
                self._challenge_out.append([c, 5])
        for item in self._challenge_out:
            self.challenges.publish(String(data=json.dumps(item[0])))
            item[1] -= 1
        self._challenge_out = [i for i in self._challenge_out if i[1] > 0]

    # -- output -------------------------------------------------------------

    def _tick(self) -> None:
        now = time.time()
        fresh = self.last_cmd is not None and now - self.last_cmd_t <= self.cmd_timeout_s
        cmd = self.last_cmd if fresh else dict(self.packs_rt.STOP_CMD)
        obs = self.scan_obs or {"stamp": 0}
        v = self.packs_rt.evaluate(self.packs, obs, cmd, now=now,
                                   heartbeat=self.heartbeat,
                                   obs_max_age_s=self.obs_max_age_s)
        self._publish(v["cmd"])
        last = self.last_verdict or {}
        if (last.get("verdict"), last.get("reason")) != (v["verdict"], v["reason"]):
            self._event({"kind": "verdict", "proposed": cmd, **v})
        self.last_verdict = v

    def _publish(self, cmd: dict) -> None:
        if self.out_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            tw = msg.twist
        else:
            msg = tw = Twist()
        tw.linear.x = float(cmd["linear"])
        tw.angular.z = float(cmd["angular"])
        self.pub.publish(msg)

    def _event(self, data: dict) -> None:
        self.events.publish(String(data=json.dumps({"t": time.time(), **data})))


def main() -> None:
    rclpy.init()
    node = SafetyFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
