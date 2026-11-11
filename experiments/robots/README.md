# Robots in simulation

Three experiments with simulated robots, run on 2026-11-11 (UTC) at commit
`664cb00` of `autonet-code/node`. They test the two seams Autonet owns in a
robot: the jurisdiction's safety packs, enforced where
commands reach the motors, and the accountability rail, made of signed
worklogs, polls, witnesses and the automatic trigger on position claims.
All three pass: 11 of 11 stages, 9 of 9 and 6 of 6.

The design is in [the whitepaper](https://github.com/autonet-code/whitepaper), under "Public Services".
This folder holds what was run and what came out.

## Setup

- **Simulator.** Gazebo (`gz sim`, headless, software rendering) in Docker,
  from `robot/sim/Dockerfile`: ROS 2 Jazzy (`ros:jazzy-ros-base`), the
  TurtleBot3 waffle model and sandbox world from `nav2_minimal_tb3_sim`,
  rosbridge, and rosbag2 with MCAP storage for the worklogs. Nav2 is not
  started: motion is sent as raw velocity commands.
- **Robots.** One TurtleBot3 in experiment 1. Three in one world in
  experiments 2 and 3 (`robot/sim/run_fleet.sh`), starting at
  (-2.0, -0.5), (-2.0, 0.5) and (-0.5, -2.0).
- **The wheels take only the filter's output.** Every command source (a
  test, a teleop, the agent's ROS connector) publishes a proposed command;
  the safety filter node (`robot/ros/safety_filter.py`) runs the required
  packs on it and publishes the result to the topic the wheels listen on.
- **The safety pack.** `robot/packs/proximity_stop.py`, an ordinary pinned
  tool: it stops forward motion with anything within 0.45 m ahead, scales
  forward speed down inside 1.0 m, and caps speed at 0.5 m/s and 1.5 rad/s.
- **The daemons** run in the test process, each against a temporary home.
  The poll rail runs over the in-process transport, not libp2p, and takes
  its rounds from a fixed anchor history instead of a chain.
- **Host.** Windows 11 with Docker 29.8.0. On a Windows host the worklog
  cannot be written through the bind mount, so it stays on the container's
  disk and the runs copy it out (`docker cp`) when they need it.

## 1. Safety packs at the actuation boundary

`scripts/local_e2e_robot.py`, one robot, 1 min 50 s.

Can a pack the jurisdiction requires keep a robot safe whatever drives it,
and does the robot stop when anything it depends on fails?

| Stage | Result |
|---|---|
| 1. Sim up: rosbridge reachable, lidar and odometry flowing | 0.93 m clear ahead |
| 2. The daemon exports the required pack; the filter loads it after checking both digests | pack `3f4c705e0453` |
| 3. Driving at an obstacle: the pack slows, then stops the robot | from 1.27 m to 0.46 m, peak speed 0.46 m/s under the 0.5 m/s cap |
| 4. Reversing away is allowed | 0.69 m clear ahead |
| 5. A tampered pack is refused and all motion stops | code digest mismatch |
| 6. Exporting the pack again restores motion | pass |
| 7. The daemon's heartbeat goes stale: the robot stops | pass |
| 8. The sponsor (the household daemon that holds judgment) is unreachable: the robot stops | pass |
| 9. A challenge: the signed link verifies and the challenge lands in the next worklog chunk; a tampered chunk and a wrong key fail | 5 chunks committed, challenge recorded |
| 10. The agent's ROS connector (ros-mcp) drives through the same filter | 31 ROS tools; 0.4 m/s forward for 6 s, closest approach 0.44 m, verdicts clip and pass |
| 11. The jurisdiction's standards, read from the Registry: a required pack the daemon lacks stops the robot; standards it can meet restore motion | pass |

## 2. The poll rail

`scripts/local_e2e_robot_poll.py`, three robots, 3 min 46 s.

Can members commit a fleet to its worklogs at moments it cannot predict, and
does every failure to answer become a report? Five member households hold
REP (100 to 500). Each round lasts 45 s; each robot gets three pollers drawn
by REP; an answer is due within 3 s; a robot whose signed links stop for
35 s is reported. The three robots are the jurisdiction's fleet, so each
witnesses the other two.

| Stage | Result |
|---|---|
| 1. Sim up: three robots, each recording a pose stream and a lidar stream | pass |
| 2. Every filter loads the jurisdiction's pack | pack `8f71713b128a` |
| 3. Continuous commitment: every robot's chain of signed links grows and verifies; witnesses and pollers hold each robot's latest link | 13, 15 and 15 links |
| 4. A full round: each robot polled by its three pollers, each poll answered by the robot and both witnesses | round 5: 3 robots by 3 pollers, each poll answered by 3 worklogs |
| 5. Each challenge appears in the chunks closed after the poll, in the robot's worklog and in both witnesses' | 27 answers, the challenge in every later worklog |
| 6. Each poller gossips one aggregate of its receipts | 3 aggregates cover the 27 receipts |
| 7. A dispute opens a countersigned window: the chunks match their links, and a tampered chunk is caught | 2 chunks over 3 links |
| 8. Thinning: the lidar stream is deleted and the pose stream still verifies | 15 lidar chunks removed, 15 pose chunks verify |
| 9. A robot goes silent, a poller stays away and a robot runs a stale pack for a round: each is reported, and nothing else is | 20 reports: the silent robot, the absent poller, the stale pack |

## 3. Scene summaries and the automatic trigger

`scripts/local_e2e_robot_scene.py`, three robots, 1 min 26 s.

Can a robot that lies about where it is be caught by the robots that could
see it, without anyone looking at images? Each robot writes a summary a
second (its pose, the objects its lidar sees in world coordinates, its pack
verdicts) and commits it in its signed links. Witness range is 1.8 m: r1 and
r2 start 1.0 m apart, and r3 is 2.1 m from r1. Three consecutive positions
a witness should have seen but found empty make a report.

| Stage | Result |
|---|---|
| 1. Sim up, each robot writing summaries in world coordinates | r1 at (-2.00, -0.50), r2 at (-2.00, 0.50), r3 at (-0.50, -2.00) |
| 2. Each robot's summaries show its neighbour where the neighbour is | r1 sees r2 0.08 m off; r2 sees r1 0.08 m off |
| 3. The summaries travel in the signed links, and every robot learns every position | 59 summaries committed |
| 4. Proximity witnesses: a poll of r1 goes to r2, which is near, and not to r3; r3 has no witnesses | r3 is 2.1 m and 2.9 m from the others |
| 5. Honest driving: r1 drives out and back, and nobody reports anything | r1 drove to x = -1.17 and back; 0 reports |
| 6. The lie: r1's daemon claims r1 stayed put while it drives away; r2, which could see the empty spot, reports it; r3, out of range, reports nothing | r2 reported r1: 14 claimed positions it could see were empty |

## What these runs do not show

- **Hardware.** Everything is simulated. Positions come from the spawn pose
  plus odometry, not GPS.
- **A real network.** The daemons share one process and the in-process
  transport. The libp2p protocol is covered only by a two-host round trip.
- **Preemption.** The filter runs packs in its own process, so a pack that
  runs too long is caught after the fact (its command is never forwarded),
  not interrupted.
- **Live injection.** Witnessing uses one sensor (lidar) and one check
  (position). A robot fed fabricated sensor data while it records is
  contradicted only where witnesses can see it. Sensor-level signing is out
  of scope until hardware supports it.
- **Judgment.** Reports are signed and gossiped, but nothing consumes them
  yet. Opening archives on a trigger, the shared perception tool,
  consolidation of many accounts into one, and judging that account against
  the standards are designed in the whitepaper and not built.
- **The sponsor.** Its reachability is stood in for
  (`ATN_ROBOT_SPONSOR_PROBE=assume`); a live run with a separate sponsor
  daemon has not been done.
- **Scale.** Every robot receives every fleet member's links; only polls and
  judgment use proximity.

## Running them again

Each needs Docker; the first run builds the simulator image (about 5 GB).

```bash
python scripts/local_e2e_robot.py
python scripts/local_e2e_robot_poll.py
python scripts/local_e2e_robot_scene.py
```

## Logs

`logs/` holds the console output of these three runs. Two warning lines in
`robot.txt` had a local cache path, replaced by `<uv cache>`. `robot_scene.txt`
has one "sync failed" line from while the fleet was starting, before its
worklog folder existed; no stage was affected.
