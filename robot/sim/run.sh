#!/usr/bin/env bash
# Container entrypoint: headless Gazebo + TurtleBot3, bridge, rosbridge,
# the Autonet safety filter and the MCAP worklog recorder.
# Mounts: /autonet (this repo, read-only), /robot (bundle/ and bags/).
set -e
source /opt/ros/jazzy/setup.bash
SIM=/opt/ros/jazzy/share/nav2_minimal_tb3_sim
export GZ_SIM_RESOURCE_PATH=$SIM/models:/opt/ros/jazzy/share
export LIBGL_ALWAYS_SOFTWARE=1
mkdir -p /robot/bundle /robot/bags

xacro -o /tmp/world.sdf $SIM/worlds/tb3_sandbox.sdf.xacro headless:=true
xacro -o /tmp/robot.sdf $SIM/urdf/gz_waffle.sdf.xacro namespace:=
gz sim -r -s --headless-rendering /tmp/world.sdf &
# Spawning waits for the world and exits once the robot exists.
ros2 run ros_gz_sim create -name turtlebot3_waffle -file /tmp/robot.sdf \
    -x -2.0 -y -0.5 -z 0.01
ros2 run ros_gz_bridge parameter_bridge --ros-args \
    -p config_file:=/autonet/robot/sim/bridge.yaml &
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090 &
python3 /autonet/robot/ros/safety_filter.py --ros-args \
    -p bundle_dir:=/robot/bundle \
    -p runtime_path:=/autonet/atn/robot/packs.py \
    -p in_topic:=cmd_vel -p stamped:=true \
    -p out_topic:=cmd_vel_safe -p out_stamped:=false -p obs_max_age_s:=0.8 &
# Worklog: every topic that matters, in 10 s MCAP chunks, on the robot's own
# disk (/var/worklog). Writing it through a Windows bind mount is unreliable,
# so on a Windows host the daemon side copies it out (docker cp) to read it.
rm -rf /var/worklog
ros2 bag record -s mcap -o /var/worklog --max-bag-duration 10     --topics /scan /odom /cmd_vel /cmd_vel_safe /safety/events /autonet/challenge &
wait -n || true
echo "run.sh: a robot process exited, shutting down" >&2
jobs -l >&2
exit 1
