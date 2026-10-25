#!/usr/bin/env bash
# Container entrypoint for the poll rail E2E (docs/robots.md, 2026-10-25):
# N TurtleBot3s (namespaces /r1../rN) in one headless Gazebo world, each with
# its own safety filter, a scene summary node (once-a-second samples into
# /robot/rI/bundle/scene.jsonl) and a two-stream MCAP worklog:
#   /var/worklog/rI/pose/  odom, commands, safety events, poll challenges, scene
#   /var/worklog/rI/scan/  lidar (the heavy stream, thinned first)
# Mounts: /autonet (this repo, read-only), /robot (rI/bundle per robot).
set -e
N=${ROBOTS:-3}
source /opt/ros/jazzy/setup.bash
SIM=/opt/ros/jazzy/share/nav2_minimal_tb3_sim
export GZ_SIM_RESOURCE_PATH=$SIM/models:/opt/ros/jazzy/share
export LIBGL_ALWAYS_SOFTWARE=1

POSES=("-2.0 -0.5" "-2.0 0.5" "-0.5 -2.0" "0.5 2.0")
rm -rf /var/worklog
xacro -o /tmp/world.sdf $SIM/worlds/tb3_sandbox.sdf.xacro headless:=true
# The robots are part of the world file, so they all exist from the start.
for i in $(seq 1 "$N"); do
  ns=r$i
  mkdir -p /robot/$ns/bundle /var/worklog/$ns
  xacro -o /tmp/$ns.sdf $SIM/urdf/gz_waffle.sdf.xacro namespace:=/$ns
  # The model's lidar visual sits below its sensor (0.121 m against 0.15 m),
  # so no lidar's scan plane crosses another robot. Give each robot a turret
  # where a real one is: a cylinder on the scan plane around the sensor (its
  # own rays start inside it).
  sed -i 's#<visual name="lidar_sensor_visual">#<visual name="lidar_turret"><pose>-0.064 0 0.15 0 0 0</pose><geometry><cylinder><radius>0.06</radius><length>0.06</length></cylinder></geometry></visual>&#' /tmp/$ns.sdf
  read -r x y <<< "${POSES[$((i-1))]}"
  sed -i "s#</world>#<include><uri>file:///tmp/$ns.sdf</uri><name>$ns</name><pose>$x $y 0.01 0 0 0</pose></include></world>#" /tmp/world.sdf
done
gz sim -r -s --headless-rendering /tmp/world.sdf &

for i in $(seq 1 "$N"); do
  ns=r$i
  read -r x y <<< "${POSES[$((i-1))]}"
  # The wheels take ONLY the filter's output, as in bridge.yaml.
  cat > /tmp/bridge_$ns.yaml <<EOF
- {ros_topic_name: "/$ns/odom", gz_topic_name: "/$ns/odom", ros_type_name: "nav_msgs/msg/Odometry", gz_type_name: "gz.msgs.Odometry", direction: GZ_TO_ROS}
- {ros_topic_name: "/$ns/scan", gz_topic_name: "/$ns/scan", ros_type_name: "sensor_msgs/msg/LaserScan", gz_type_name: "gz.msgs.LaserScan", direction: GZ_TO_ROS}
- {ros_topic_name: "/$ns/cmd_vel_safe", gz_topic_name: "/$ns/cmd_vel", ros_type_name: "geometry_msgs/msg/Twist", gz_type_name: "gz.msgs.Twist", direction: ROS_TO_GZ}
EOF
  if [ "$i" = 1 ]; then
    echo '- {ros_topic_name: "/clock", gz_topic_name: "/clock", ros_type_name: "rosgraph_msgs/msg/Clock", gz_type_name: "gz.msgs.Clock", direction: GZ_TO_ROS}' >> /tmp/bridge_$ns.yaml
  fi
  ros2 run ros_gz_bridge parameter_bridge --ros-args -r __node:=bridge_$ns \
      -p config_file:=/tmp/bridge_$ns.yaml &
  python3 /autonet/robot/ros/safety_filter.py --ros-args -r __ns:=/$ns \
      -p bundle_dir:=/robot/$ns/bundle \
      -p runtime_path:=/autonet/atn/robot/packs.py \
      -p in_topic:=cmd_vel -p stamped:=true \
      -p out_topic:=cmd_vel_safe -p out_stamped:=false -p obs_max_age_s:=0.8 &
  python3 /autonet/robot/ros/scene_summary.py --ros-args -r __ns:=/$ns \
      -p bundle_dir:=/robot/$ns/bundle -p runtime_path:=/autonet/atn/robot/scene.py \
      -p spawn_x:="$x" -p spawn_y:="$y" -p spawn_yaw:=0.0 &
  ros2 bag record -s mcap -o /var/worklog/$ns/pose/worklog --max-bag-duration 10 \
      --topics /$ns/odom /$ns/cmd_vel /$ns/cmd_vel_safe /$ns/safety/events \
               /$ns/autonet/challenge /$ns/autonet/scene &
  ros2 bag record -s mcap -o /var/worklog/$ns/scan/worklog --max-bag-duration 10 \
      --topics /$ns/scan &
done
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090 &
wait -n || true
echo "run_fleet.sh: a robot process exited, shutting down" >&2
jobs -l >&2
exit 1
