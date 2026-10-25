"""Scene summaries and the automatic trigger (docs/robots.md, Phase B)."""
from __future__ import annotations

import asyncio
import json
import math

from eth_account import Account

from atn.robot import poll as P
from atn.robot import scene as S
from atn.robot.node import LocalNetwork, PollNode, PollParams
from atn.robot.scene_source import SceneSource

# A robot body is a circle of this radius; pillars are larger circles.
BODY = 0.15


def scan_from(pose, discs, n=360, rmax=8.0):
    """Ray-cast a 360-ray lidar from ``pose`` against circles (x, y, r)."""
    x, y, yaw = pose
    ranges = []
    for i in range(n):
        a = yaw + i * 2 * math.pi / n
        dx, dy = math.cos(a), math.sin(a)
        best = math.inf
        for cx, cy, r in discs:
            fx, fy = x - cx, y - cy
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - r * r
            disc = b * b - c
            if disc >= 0:
                t = -b - math.sqrt(disc)
                if t > 0.05:
                    best = min(best, t)
        ranges.append(best if best <= rmax else math.inf)
    return {"ranges": ranges, "angle_min": 0.0, "angle_increment": 2 * math.pi / n,
            "range_min": 0.05, "range_max": rmax}


def sample(t, pose, others, pillars=()):
    discs = [(o[0], o[1], BODY) for o in others] + list(pillars)
    return S.make_sample(t=t, pose=list(pose), v=0.0, scan=scan_from(pose, discs))


def test_clusters_find_a_robot_and_cut_walls():
    s = sample(0, (0, 0, 0), [(1.5, 0.0)])
    assert any(math.hypot(o[0] - 1.5, o[1]) < 0.2 for o in s["objects"])
    # A long run (a wall of touching discs) comes back in short pieces.
    wall = [(2.0, y / 10, 0.06) for y in range(-15, 16)]
    s = S.make_sample(t=0, pose=[0, 0, 0], v=0, scan=scan_from((0, 0, 0), wall))
    assert len(s["objects"]) >= 4 and all(o[2] <= 0.55 for o in s["objects"])


def test_view_is_blocked_behind_a_pillar_and_limited_by_range():
    w = sample(0, (0, 0, 0), [], pillars=[(1.0, 0.0, 0.3)])
    assert S.view(w, (2.5, 0.0), range_m=3.0, margin=0.35) == "blocked"
    assert S.view(w, (0.0, 2.0), range_m=3.0, margin=0.35) == "visible"
    assert S.view(w, (0.0, 5.0), range_m=3.0, margin=0.35) == "out_of_range"


def _track(points, t0=100.0):
    return [{"t": t0 + i, "pose": [x, y, 0.0], "v": 0.0} for i, (x, y) in enumerate(points)]


def test_honest_claims_are_seen():
    truth = [(1.0 + 0.1 * i, -1.0) for i in range(6)]
    own = [sample(100.0 + i, (0, 0, 0), [p]) for i, p in enumerate(truth)]
    checks = S.check_claims(_track(truth), own)
    assert len(checks) == 6 and all(c["seen"] for c in checks)
    assert S.conflict_runs(checks) == []


def test_a_fabricated_track_is_a_conflict():
    truth = [(1.0 + 0.3 * i, -1.0) for i in range(6)]       # it drove off
    claimed = [(1.0, -1.0)] * 6                               # it says it stayed
    own = [sample(100.0 + i, (0, 0, 0), [p]) for i, p in enumerate(truth)]
    runs = S.conflict_runs(S.check_claims(_track(claimed), own))
    assert runs and len(runs[0]) >= 3


def test_a_claim_hidden_behind_a_pillar_is_no_evidence():
    pillar = [(1.0, 0.0, 0.3)]
    own = [sample(100.0 + i, (0, 0, 0), [], pillars=pillar) for i in range(6)]
    checks = S.check_claims(_track([(2.5, 0.0)] * 6), own)
    assert checks == []


def test_one_missed_frame_is_not_a_conflict():
    truth = [(1.0, -1.0)] * 6
    own = [sample(100.0 + i, (0, 0, 0), [] if i == 2 else [p]) for i, p in enumerate(truth)]
    assert S.conflict_runs(S.check_claims(_track(truth), own)) == []


def test_scene_source_reads_appended_lines_and_skips_a_half_line(tmp_path):
    import time
    t = round(time.time())
    f = tmp_path / "scene.jsonl"
    src = SceneSource(f)
    f.write_text(json.dumps({"t": t, "pose": [0, 0, 0]}) + "\n" + '{"t": %d, "po' % (t + 1))
    assert [s["t"] for s in src.between(0, t + 10)] == [t]
    with f.open("a") as fh:
        fh.write('se": [0, 0, 0]}\n')
    assert [s["t"] for s in src.between(0, t + 10)] == [t, t + 1]
    assert [s["t"] for s in src.between(t, t + 10)] == [t + 1]


# -- in the poll rail ---------------------------------------------------------

class FakeScene:
    """A robot's own samples, generated from where everything truly is."""

    def __init__(self, world, me, clock, lie=None):
        self.world, self.me, self.clock, self.lie = world, me, clock, lie

    def between(self, t0, t1):
        out = []
        t = math.floor(t0) + 1
        while t <= t1:
            pos = self.world(t)
            others = [p for r, p in pos.items() if r != self.me]
            s = sample(t, (*pos[self.me], 0.0), others)
            if self.lie is not None:
                s["pose"] = [*self.lie, 0.0]
            out.append(s)
            t += 1
        return out


def test_witness_reports_a_robot_lying_about_where_it_is(tmp_path):
    now = [1000.0]
    clock = lambda: now[0]  # noqa: E731
    accts = [Account.create() for _ in range(3)]
    robots = [a.address.lower() for a in accts]
    keys = {a.address.lower(): a.key.hex() for a in accts}
    # r0 drives away along x; r1 and r2 stand still nearby.
    world = lambda t: {robots[0]: (0.0 + 0.3 * max(0.0, t - 1005), 0.0),  # noqa: E731
                       robots[1]: (0.0, 1.2), robots[2]: (0.0, -1.2)}
    net = LocalNetwork()
    rnd = P.Round(index=1, start=900.0, seed="ab" * 32, period_s=10_000.0,
                  members=tuple(P.Member(r, r, 0) for r in robots))
    nodes = {}
    for i, r in enumerate(robots):
        from atn.robot import worklog as W
        bag = tmp_path / r / "bags"
        node = PollNode(
            address=r, private_key=keys[r], transport=net.transport(r),
            params=PollParams(k=0, percentile=0), round_at=lambda t: rnd,
            roster=lambda x: [y for y in robots if y != x.lower()], robots=lambda: robots,
            state_dir=tmp_path / r / "state",
            worklog=W.Worklog(bag, tmp_path / r / "wl", r, keys[r]),
            scene=FakeScene(world, r, clock, lie=(0.0, 0.0) if i == 0 else None),
            clock=clock)
        nodes[r] = node
        net.add(node)

    async def run():
        for step in range(30):
            now[0] += 1
            if step % 5 == 4:
                from tests.atn.test_robot_poll import _chunk
                for r in robots:
                    _chunk(tmp_path / r / "bags", "pose", step, b"x%d" % step, step)
            for n in nodes.values():
                await n.step()
    asyncio.run(run())
    reps = [g for g in net.gossip_log if g.get("type") == P.REPORT_TYPE]
    assert reps and {g["kind"] for g in reps} == {"scene_conflict"}
    assert {g["robot"] for g in reps} == {robots[0]}
    assert {g["reporter"] for g in reps} == {robots[1], robots[2]}
    # The honest robots' claims raised nothing, and positions are tracked.
    assert nodes[robots[1]].positions[robots[2]] == (0.0, -1.2)
