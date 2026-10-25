# Robots: Autonet agents with a body

Status: MVP built (2026-10-19), sim only, E2E 10/10. Design decided in conversation with the
owner; this file is the ledger. Nothing here touches contracts.

## The three layers

A robot runs at three speeds, and Autonet owns only two seams of them.

| Layer | Speed | Who owns it |
|-------|-------|-------------|
| Body control (proprioception, motor policies) | 100 to 1000 Hz | Off the shelf: ROS 2, `ros2_control`, Nav2, open VLAs |
| Directives (safety packs) | Same rate as the commands they filter | Autonet: governed, hash-pinned packs enforced on the robot |
| Agent (goals, judgment, escalation) | Seconds | Autonet: the existing agent loop, unchanged |

The agent sets goals and calls skills. It never closes a motor loop.

## Decisions

1. **The robot runs its own daemon, as a sponsored dependent of the household
   daemon** (`docs/sponsored_inference.md`). All judgment goes through the
   sponsor: the sponsor agent is the cognitive backstop and the escalation
   target. The household is the liable entity. The robot holds no provider
   credentials, so a compromised robot has no key to steal.
2. **Sponsor unreachable means packs-only safe stop.** No judgment available,
   no motion: the robot stops and holds.
3. **Directives are safety packs, and a pack is an ordinary pinned tool.** It
   is authored, bonded, reviewed and adopted like any tool, so a pack that
   fails and leads to harm slashes its author's bond. The packs a jurisdiction
   requires are its **standards**.
4. **Packs run on the robot, below the agent, at the actuation boundary.**
   Every command source (Nav2, a teleop, the agent through its connector) only
   *proposes* motion. Only the filter's output reaches the motors.
5. **Fail closed.** A missing or tampered required pack, a pack that raises or
   runs too long, stale sensor data, or a stale sponsor heartbeat all produce
   a stop.
6. **Polling is for robots in shared space only.** A DAO (or anyone the robot
   answers to) can send a random challenge. The robot answers before a deadline
   with a signed statement over its recent worklog (rosbag2 MCAP chunks), the
   challenge and the active pack digests. Faking that worklog in time is harder
   than sending the real one, and witnesses sharing the space can cross-check
   it. Private digital daemons are never polled: what an agent does in the
   privacy of its own daemon is not the network's business. Who polls, when,
   and how witness records combine are set by the 2026-10-25 decision below.
7. **Identity in public space**: household and robot 0x addresses on the body
   (printed, QR) plus a signed broadcast beacon, Remote ID style. Not in the
   MVP.
8. **Public services** are DAOs under the state DAO (fractal block), running
   their own daemons and fleets. Subordinates are funded by transfer proposals
   or tranche drips with a halt vote; trustless projects are for peers. Not in
   the MVP.

## The seam: ROS 2

ROS 2 is the de facto standard for open robot stacks (not universal: Tesla and
Figure run their own). Autonet plugs in at three points:

- **Skills as tools.** `ros-mcp-server` (PyPI `ros-mcp`) exposes topics,
  services and actions over rosbridge as MCP tools. The daemon takes it as an
  ordinary MCP connector. No new daemon code.
- **Safety filter node** (`robot/ros/safety_filter.py`). Subscribes to the
  proposed command topic and the sensors, runs the required packs, publishes
  the filtered command to the topic the motors listen on, and publishes every
  verdict on `/safety/events`. The pack logic itself is plain stdlib Python
  (`atn/robot/packs.py`) so the same code runs in the daemon and on the robot.
- **Worklog.** `ros2 bag record` with MCAP storage, split into short chunks.
  The challenge response hashes the chunks that overlap the window.

## Pack contract

A pack is a pinned tool whose code defines

```python
def check(obs: dict, cmd: dict) -> dict:
    # obs: {"stamp": float, "scan": {"front_min": float, ...}, ...}
    # cmd: {"linear": float, "angular": float}
    return {"verdict": "pass" | "clip" | "stop", "cmd": {...}, "reason": str}
```

and also works as a normal tool (JSON `{obs, cmd}` on stdin, verdict JSON on
stdout), so agents can test and review it with `use_tool`. A pack declares no
capabilities: no network, no files, no subprocesses. The runtime refuses one
that does.

Composition: every required pack runs on every command. `stop` beats `clip`
beats `pass`; clips apply in order.

Pinning: the daemon exports each required pack as `<digest>.json` (the signed
manifest) plus `<code_digest>.py`. The filter recomputes both sha256 digests
before loading. The required set lives in the robot daemon's config for the
MVP (`robot.required_packs`); reading it from the jurisdiction's Registry
(`jurisdiction.<id>.robot_standards`) is a follow-up.

## MVP

No hardware. Everything runs against a simulated TurtleBot3 in Gazebo,
headless, in Docker (`robot/sim/`). Nav2 is not started: the agent's skills
for the MVP are raw topic publishes through ros-mcp.

| Piece | Where |
|-------|-------|
| Pack runtime (stdlib only) | `atn/robot/packs.py` |
| Reference pack: proximity stop + speed cap | `robot/packs/proximity_stop.py` |
| ROS 2 filter node | `robot/ros/safety_filter.py` |
| Worklog statement, signing, verification | `atn/robot/worklog.py` |
| Daemon side (export, heartbeat, attest) | `atn/robot/service.py`, `robot:` in config.yaml, WS `robot_status` / `robot_export_packs` / `robot_attest` (owner only) |
| Sim image and entrypoint | `robot/sim/Dockerfile`, `run.sh`, `bridge.yaml` |
| Unit tests | `tests/atn/test_robot_packs.py` |
| End to end | `scripts/local_e2e_robot.py` |

Config on the robot's daemon:

```yaml
robot:
  enabled: true
  agent_id: rover            # its key signs attestations
  required_packs: [proximity_stop]
  bundle_dir: /robot/bundle  # read by the filter node
  bag_dir: /robot/bags       # MCAP worklog chunks
```

The E2E (`python scripts/local_e2e_robot.py`, needs Docker) passes 10/10:
sim up; pack exported and loaded; driving at an obstacle is slowed then
stopped (from 1.27 m to 0.46 m, peak 0.46 m/s under a 0.5 cap); reversing
away is allowed; a tampered pack is refused and nothing moves; re-export
restores motion; a stale daemon heartbeat stops the robot; an unreachable
sponsor stops it; a challenge answer verifies, and a tampered chunk, a wrong
key and a late answer all fail; the agent's ros-mcp connector pushing forward
for 6 s is clipped and never gets closer than the stop distance.

Known limits of the MVP:

- The filter runs packs in-process, so an over-budget pack is caught after the
  fact (its command is never forwarded, but it cannot be preempted).
- Sensor age is a filter parameter: 0.5 s by default, 0.8 s in the sim
  because its lidar runs at 5 Hz with occasional missed frames. Brief stops
  from a missed frame are the intended behavior.
- On a Windows host the worklog cannot be written through the bind mount
  (bag rollover fails), so it stays on the container's disk and the E2E copies
  it out with `docker cp` before a challenge. On a real robot the daemon reads
  its own disk.
- The challenge is not yet folded into the worklog itself (publishing it on a
  topic so the next chunk carries it). The deadline is what makes faking hard
  today.
- Sponsor reachability is "the named sponsor is advertising on p2p", from the
  sponsored provider's discovery. The E2E stands it in with
  `ATN_ROBOT_SPONSOR_PROBE=assume`; a live two-daemon sponsor run is not done.
- The robot agent in the E2E is an `external` agent (it only signs). A real
  robot agent would run on `rpb` through its sponsor.

Out of the MVP: beacon, sanctions list, TPM quotes, witness cross-checks,
Registry-backed standards, public-service DAOs, Nav2 skills. Polls and
witnesses are designed below (2026-10-25); the poll rail (Phase A) is built,
consolidation (Phase B) is not. The worklog statement of the MVP is replaced
by the signed link chain: `robot_attest` now commits a link carrying the
challenge.

## Decision (2026-10-25): polls, witnesses and physical consensus

Design agreed with the owner. Phase A (the poll rail) is fully specified;
Phase B (consolidation) has its shape decided and may still move in detail.
Numbers marked *provisional* are defaults for the jurisdiction to set. They
are relative (percentiles, epochs, coverage ratios), never nominal amounts,
so they do not need updating as the jurisdiction grows.

### What a poll proves, and what it cannot

A poll forces a robot to commit to its worklog at a moment it could not
predict. The commitment proves the record was not changed **afterwards**. It
cannot prove the record was **true** when written: a compromised robot can
fabricate data before it hashes it, or its handlers can feed it generated
sensor data live. So:

- A robot's own record can incriminate it but never clears it against
  contradicting witnesses.
- Frequent commitments defeat after-the-fact forgery (the common case: an
  unplanned incident, then an attempt to cover it). Only witnesses and
  sensor-level signing defend against premeditated live injection.

### Phase A: the poll rail

1. **Who polls.** Each epoch, each robot gets its own `k` pollers (*provisional*
   3), drawn REP-weighted from members above a REP percentile (*provisional*
   the 90th). The seed is an anchored epoch root from at least two anchors
   back (so a robot's owner cannot steer it through the events they submit)
   plus the robot's address. The draw is public and anyone can recompute it.
   Extra daemons gain nothing: weight is soulbound REP, not peer count.
2. **When they poll.** Each poller signs `(epoch root, robot, slot)` with its
   own key. Ethereum signatures are deterministic (RFC 6979), so the
   signature fixes both the poll time inside the epoch and the challenge.
   Only the poller can compute it in advance; afterwards anyone can verify
   neither was cherry-picked. *Provisional*: one poll per poller per epoch.
3. **The poll.** The poller sends the challenge to the robot and to its
   witnesses at once. Each writes it onto a worklog topic, so it lands in
   each one's next chunk and gives every log a shared instant without
   trusting anyone's clock. Each answers before the deadline with a signed
   statement.
4. **The statement** signs a merkle root, not a flat list. Leaves are
   `(chunk, topic stream)`, so old data can be thinned stream by stream
   (drop camera early, keep pose, commands and pack verdicts for years) and
   every kept leaf still verifies against the signed root. Each statement
   carries the previous statement's root: the whole worklog is one chain,
   and no chunk can be dropped or inserted between polls. The chunk still
   being written is covered by the next commitment.
5. **Continuous commitment.** Between polls, the robot sends its new chain
   root to its witnesses and current pollers every time a chunk closes
   (seconds, about 100 bytes per peer, direct, no gossip). A late or
   missing root is itself evidence, like a missed heartbeat. The forgery
   window after an incident shrinks to one chunk length.
6. **Countersigning.** The poller checks the statement (challenge, deadline
   by its own clock, key, required packs, chain continuity) and countersigns
   it with its receive time. The robot archives its raw chunks with the
   countersigned statements; the poller keeps nothing. Pollers aggregate
   their receipts into one merkle root per epoch and gossip that, one
   message per poller per epoch.
7. **Retention**, counted in epochs (*provisional* values set by the
   jurisdiction). A witness keeps its record for a shorter period than the
   robot keeps its own. Storage is the robot's own disk: swap or upload
   when full.
8. **Exceptions are the only consensus traffic.** A late, invalid or
   chain-breaking answer, a missing commitment, or a drawn poller that
   never polled becomes a report. Passed polls cost one hash each inside
   an aggregate.

### Phase B: from many perspectives to one account

1. **Scene summaries.** With each committed chunk, every party also commits
   a small structured summary from its on-board perception: the entities it
   tracks (with positions) and the events it noticed (contact, near miss, a
   robot entering a zone). A robot adds its own pack verdicts (every clip and
   stop). Summaries carry no images, so nothing about bystanders leaks, and
   they are a merkle leaf like any stream.
2. **The automatic trigger.** Matching summaries across parties is cheap and
   close to deterministic. A conflict is any of: a witness logs an event
   involving robot A that A's log lacks; a witness tracks A away from where
   A says it was (position is never the matter disputed, but a fabricated
   record drifts from what others saw); A was in a witness's view and the
   witness logged nothing (silence where others saw something is itself a
   mismatch, which catches witnesses who never notice).
3. **Subjective reports.** A witness's agent or a human may also report
   conduct the summaries cannot express, with a stake lost if the report
   fails.
4. **A trigger opens the archives.** On a conflict or a staked report, the
   robot and its witnesses produce their raw chunks; their hashes must match
   the countersigned roots. A countersigned window that cannot be produced
   counts against whoever fails to produce it (the flight-recorder rule).
   Raw data is opened only to the consolidators drawn for the case (same
   lottery as pollers) and to any challenger who stakes; only the resulting
   account is public. Bystanders never consented to being recorded.
   **Anonymized at the source.** Blurring after the fact would break the
   committed hashes, so each camera records two streams, each its own merkle
   leaf: raw, and a copy with faces and plates blurred on board. The blurred
   stream is the default evidence and verifies against the root by itself;
   most conduct questions need no one's identity. The raw stream stays
   sealed on the robot and opens only by DAO vote (as a court unseals
   evidence). The drawn consolidators may re-run the pinned blur tool on
   the sealed raw stream to confirm the blurred copy is faithful (that
   nothing incriminating was blurred away). Blurred streams can be retained
   longer than raw ones. A jurisdiction may also choose blur-only recording,
   trading evidence for privacy. The network never needs to know who a
   bystander is to judge what a robot did.
5. **A common description.** Raw sensor data from different viewpoints is
   not comparable. Each party's archive runs through the same governed,
   pinned perception tool (authored, bonded, reviewed and adopted like a
   safety pack), giving an account in world coordinates: entities, tracks,
   and events such as contact or entering a restricted zone.
6. **Consolidation.** Entities and events are matched across accounts. A
   fact stands when most observers who could have seen it (in range and in
   view, by their own pose) confirm it, and is struck when they contradict
   it. Too few capable observers gives "unknown", never guilt. The robot's
   own account is one account, and must additionally be consistent with its
   own raw log across sensors (video, lidar, IMU, odometry, GPS, motor
   currents, pack verdicts at their own rates) and with what its packs did.
   Consolidation is paid work and is checked by re-running it: the pinned
   tool and the archived data fix the inputs. Vision on GPUs is not
   bit-identical across hardware, so re-runs agree on facts within
   tolerances, not on hashes; a challenger who posts a contradicting account
   is resolved like a review, and the wrong side loses its stake.
   **Who pays (owner's decision, 2026-10-25): justice is a public service.**
   A justice DAO under the jurisdiction (decision 8) is funded from the
   treasury's share of service fees by tranche drips with a halt vote, and
   pays the drawn consolidators per case, so nobody is priced out of
   reporting harm. Loser pays: an upheld violation reimburses the case from
   the offender's slashed bond; a failed staked report is covered by the
   reporter's stake. Automatic triggers (summary conflicts) need no stake
   and are carried by the public funding, as a prosecutor acts without a
   private complainant.
7. **Judgment.** The jurisdiction's standards are predicates over the agreed
   account (the same kind of rule a safety pack enforces). A violation
   becomes a report; the DAO decides any sanction. Inconclusive cases go to
   a human vote, with the raw data as evidence.

### Physical-layer security

In physical space the threat is not only inside the jurisdiction: anyone can
spoof GPS, blind a sensor or tamper with a camera, and fixed infrastructure
is a witness like any other, equally corruptible. The chain secures the
economic layer; the physical layer is secured by an honest majority of
independent observers in each place.

- **Independence by household.** Witnesses are counted per household (the
  proven owner wallet collapse), so one owner's ten robots are one witness.
  A closed group of colluding witnesses is the main attack surface.
- **Coverage standards, per territory.** A jurisdiction states a coverage
  standard (*provisional* example: every moment of robot activity in shared
  space seen by at least 3 independent households). The fleet size a
  territory needs (a cruise ship, a village, a city) follows from its
  area, sensor range and lines of sight, and coverage is measured, not
  estimated. Each robot carries a public coverage figure.
- **Below standard, degrade safely.** In an under-covered territory, facts
  come out "unknown" and the safety packs move to more conservative limits
  until coverage recovers: the same fail-closed rule as a stale sensor.
- **Surprise physical audits.** The poll lottery can also dispatch an outside
  witness (a patrol robot or drone from a public-service DAO or another
  household) at a time the target cannot predict, weighted toward thinly
  covered areas. A closed group can never be sure nobody is watching.
- **Sensor-level signing** (secure elements, C2PA-style) and TPM quotes of
  the recording pipeline are the defence against live injection. They need
  hardware support and stay out of scope until then. The same hardware path
  would put the blur on a separate compute unit attached to the camera,
  outside the robot's main operating system, so raw frames never reach it.

### Build order

Phase A is **built** (2026-10-25). Phase B is next, with a staged violation:
one robot breaks a rule while the others watch.

| Piece | Where |
|-------|-------|
| Signing (deterministic, low-s only) and merkle proofs | `atn/robot/crypto.py` |
| Worklog chain: per-stream leaves, signed links, thinning | `atn/robot/worklog.py` |
| Poller draw, signature-timed polls, receipts, aggregates, reports | `atn/robot/poll.py` |
| One daemon's roles: robot, witness, poller, watcher | `atn/robot/node.py` |
| In the daemon: rounds from anchors, rosters from the Registry, libp2p | `atn/robot/rail.py`, `robot_poll:` in config.yaml (off by default) |
| Wire | `/autonet/robot/1.0.0` (`nodes/common/p2p.py`), gossip topic `autonet/robot/1` |
| Chain inputs | `read_voice_state(..., anchor_index=i)` gives `agent_rep` and `territories` (Registry `territory.<name>` = `{"robots": [...]}`) |
| Challenge into the worklog | daemon writes `challenge.json` in the bundle; the filter publishes `autonet/challenge` |
| Three-robot sim | `robot/sim/run_fleet.sh` (pose and scan streams per robot) |
| Unit tests | `tests/atn/test_robot_poll.py` |
| End to end | `scripts/local_e2e_robot_poll.py` |

Rounds: round `r` starts at `T0 + r * period_s` (`T0` = the first anchor's
timestamp) and is pinned to the anchor in force at its start; polling begins
once two anchors precede a round, since the seed is the root two anchors
back. A drawn poller polls at its signature-fixed time; a robot refuses a
poll that arrives early, and at a round's end reports drawn pollers that
never came. Watchers (witnesses and the round's pollers) report a robot
whose links stop for `commit_gap_s`.

The E2E (`python scripts/local_e2e_robot_poll.py`, needs Docker) passes 9/9:
three robots record two streams each; every chain grows and verifies and
every witness holds each robot's head; in a full round all 9 polls are
answered by the subject and both witnesses (27 countersigned answers); every
challenge is found in the chunks each of the three worklogs closed after the
poll; 3 aggregates cover all 27 receipts; a dispute's chunks match their
links and a tampered chunk is caught; pruning the scan stream leaves the pose
leaves and the chain verifying; a silent robot and an absent poller produce
missing-commit, unanswered-poll and absent-poller reports and nothing else.

Known limits of Phase A:

- The E2E runs the daemons in one process over the in-process transport.
  The libp2p protocol is covered by a two-host round trip, not yet by a
  multi-daemon run.
- The challenge-in-the-next-chunk check runs at dispute time, from the
  archive. Pollers hold hashes only and cannot check it live.
- Whether the robot's packs match the jurisdiction's required set is not
  checked by pollers yet (`verify_answer` takes the set; the daemon does not
  pass it until standards move to the Registry).
- Reports are signed and gossiped; nothing consumes them yet (Phase B and
  the accountability rail).
