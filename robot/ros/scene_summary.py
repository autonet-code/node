"""ROS 2 scene summary node (docs/robots.md, 2026-10-25, Phase B).

Once a second it turns the robot's latest lidar scan and pose into a scene
sample (``atn/robot/scene.py``, loaded by path: stdlib only, and this node
runs under the robot's ROS Python). The pose is in world coordinates: GPS
on a real robot; here the spawn pose composed with odometry. Each sample is
appended to ``<bundle_dir>/scene.jsonl`` for the daemon, which commits it in
the robot's signed links, and published on ``autonet/scene`` so the worklog
records it too. Safety-pack verdicts since the previous sample ride along.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def _load(path: str):
    spec = importlib.util.spec_from_file_location("atn_robot_scene", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atn_robot_scene"] = mod
    spec.loader.exec_module(mod)
    return mod


class SceneSummary(Node):
    def __init__(self) -> None:
        super().__init__("autonet_scene_summary")
        p = self.declare_parameter
        self.bundle_dir = p("bundle_dir", "/robot/bundle").value
        self.scene = _load(p("runtime_path", "/autonet/atn/robot/scene.py").value)
        # Where odometry's origin sits in the world (the spawn pose), and the
        # lidar's offset ahead of the base centre (TurtleBot3 waffle: -0.064 m).
        self.spawn = (float(p("spawn_x", 0.0).value), float(p("spawn_y", 0.0).value),
                      float(p("spawn_yaw", 0.0).value))
        self.lidar_x = float(p("lidar_x", -0.064).value)
        self.odom = None
        self.scan = None
        self.verdicts = {"clip": 0, "stop": 0}
        self.create_subscription(LaserScan, "scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        self.create_subscription(String, "safety/events", self._on_event, 10)
        self.pub = self.create_publisher(String, "autonet/scene", 10)
        self.create_timer(1.0, self._tick)
        os.makedirs(self.bundle_dir, exist_ok=True)
        self.get_logger().info(f"scene summary -> {self.bundle_dir}/scene.jsonl")

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan = {"ranges": [float(r) for r in msg.ranges],
                     "angle_min": float(msg.angle_min),
                     "angle_increment": float(msg.angle_increment),
                     "range_min": float(msg.range_min),
                     "range_max": min(float(msg.range_max), 8.0)}

    def _on_odom(self, msg: Odometry) -> None:
        self.odom = msg

    def _on_event(self, msg: String) -> None:
        try:
            e = json.loads(msg.data)
        except ValueError:
            return
        if e.get("kind") == "verdict" and e.get("verdict") in self.verdicts:
            self.verdicts[e["verdict"]] += 1

    def _world_pose(self) -> list[float]:
        o = self.odom.pose.pose
        q = o.orientation
        oyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        sx, sy, syaw = self.spawn
        x = sx + math.cos(syaw) * o.position.x - math.sin(syaw) * o.position.y
        y = sy + math.sin(syaw) * o.position.x + math.cos(syaw) * o.position.y
        yaw = math.atan2(math.sin(syaw + oyaw), math.cos(syaw + oyaw))
        return [x, y, yaw]

    def _tick(self) -> None:
        if self.odom is None or self.scan is None:
            return
        base = self._world_pose()
        # The scan is taken at the lidar, slightly behind the base centre.
        lidar = [base[0] + self.lidar_x * math.cos(base[2]),
                 base[1] + self.lidar_x * math.sin(base[2]), base[2]]
        tw = self.odom.twist.twist
        sample = self.scene.make_sample(
            t=time.time(), pose=lidar, v=math.hypot(tw.linear.x, tw.linear.y),
            scan=self.scan, verdicts=self.verdicts)
        sample["pose"] = [round(base[0], 4), round(base[1], 4), round(base[2], 4)]
        self.verdicts = {"clip": 0, "stop": 0}
        line = json.dumps(sample, separators=(",", ":"))
        with open(os.path.join(self.bundle_dir, "scene.jsonl"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.pub.publish(String(data=line))


def main() -> None:
    rclpy.init()
    node = SceneSummary()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
