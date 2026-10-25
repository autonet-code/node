"""Scene summaries and the automatic trigger (docs/robots.md, 2026-10-25,
Phase B steps 1 and 2).

Stdlib only: the robot's ROS node loads this file by path to build its
samples, and the daemon imports it to compare them.

A **sample** is what one robot perceived at one moment, once a second, in
world coordinates (GPS on a real robot; spawn pose plus odometry in the sim):

    {"t": unix seconds, "pose": [x, y, yaw], "v": speed m/s,
     "objects": [[x, y, width], ...],   # lidar clusters, never images
     "sectors": [range, ...],           # nearest return per 10 degree sector
     "verdicts": {"clip": n, "stop": n}} # its safety packs since the last one

Samples carry no pixels, so nothing about bystanders leaks. A robot commits
its samples inside its signed worklog links.

The trigger runs on every witness that receives another robot's link: for
each of the subject's samples, take the witness's own sample from the same
moment. If the subject claims to be within sensing range, and the witness's
sectors show a clear line of sight to that spot (nothing nearer in that
direction), the witness must have seen an object there. A run of such
misses is a ``scene_conflict``. Where the view is blocked or out of range
the witness has no evidence either way, and nothing is reported.
"""
from __future__ import annotations

import math
from typing import Any

N_SECTORS = 36


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def clusters(ranges: list[float], angle_min: float, angle_inc: float,
             range_min: float, range_max: float, *, gap: float = 0.15,
             min_points: int = 2, piece: float = 0.5) -> list[tuple[float, float, float]]:
    """Group consecutive lidar returns into objects: (bearing, range, width)
    in the robot's frame. A run longer than ``piece`` (a wall, or a robot
    standing against one) is cut into pieces of at most that length, so an
    object touching scenery still shows up near where it is."""
    pts = []
    for i, r in enumerate(ranges):
        if r is not None and math.isfinite(r) and range_min <= r <= range_max:
            a = angle_min + i * angle_inc
            pts.append((i, a, r, r * math.cos(a), r * math.sin(a)))
    groups: list[list[tuple]] = []
    for p in pts:
        if groups and p[0] == groups[-1][-1][0] + 1 and \
                math.hypot(p[3] - groups[-1][-1][3], p[4] - groups[-1][-1][4]) <= gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    # The scan wraps around: join the last group onto the first.
    n = len(ranges)
    if len(groups) > 1 and groups[0][0][0] == 0 and groups[-1][-1][0] == n - 1 and \
            math.hypot(groups[0][0][3] - groups[-1][-1][3],
                       groups[0][0][4] - groups[-1][-1][4]) <= gap:
        groups[0] = groups.pop() + groups[0]
    out = []
    for g in groups:
        if len(g) < min_points:
            continue
        pieces, cur = [], [g[0]]
        for p in g[1:]:
            if math.hypot(p[3] - cur[0][3], p[4] - cur[0][4]) > piece:
                pieces.append(cur)
                cur = []
            cur.append(p)
        pieces.append(cur)
        for pc in pieces:
            width = math.hypot(pc[0][3] - pc[-1][3], pc[0][4] - pc[-1][4])
            x = sum(p[3] for p in pc) / len(pc)
            y = sum(p[4] for p in pc) / len(pc)
            out.append((math.atan2(y, x), math.hypot(x, y), width))
    return out


def sectors(ranges: list[float], angle_min: float, angle_inc: float,
            range_min: float, range_max: float, n: int = N_SECTORS) -> list[float]:
    """Nearest return per sector, robot frame, sector 0 centred straight
    ahead and counting counterclockwise. No return = ``range_max``."""
    out = [float(range_max)] * n
    width = 2 * math.pi / n
    for i, r in enumerate(ranges):
        if r is None or not math.isfinite(r) or not range_min <= r <= range_max:
            continue
        a = _wrap(angle_min + i * angle_inc)
        k = int(((a + width / 2) % (2 * math.pi)) // width) % n
        out[k] = min(out[k], float(r))
    return out


def to_world(pose: list[float], bearing: float, rng: float) -> tuple[float, float]:
    x, y, yaw = pose
    return x + rng * math.cos(yaw + bearing), y + rng * math.sin(yaw + bearing)


def make_sample(*, t: float, pose: list[float], v: float, scan: dict[str, Any],
                verdicts: dict[str, int] | None = None) -> dict[str, Any]:
    """One sample from a pose and a raw scan (``ranges``, ``angle_min``,
    ``angle_increment``, ``range_min``, ``range_max``)."""
    args = (scan["ranges"], scan["angle_min"], scan["angle_increment"],
            scan["range_min"], scan["range_max"])
    objs = []
    for bearing, rng, width in clusters(*args):
        x, y = to_world(pose, bearing, rng)
        objs.append([round(x, 3), round(y, 3), round(width, 3)])
    return {"t": round(float(t), 3), "pose": [round(float(p), 4) for p in pose],
            "v": round(float(v), 3), "objects": objs,
            "sectors": [round(s, 3) for s in sectors(*args)],
            "verdicts": dict(verdicts or {})}


def _nearest(samples: list[dict], t: float, max_dt: float) -> dict | None:
    best = None
    for s in samples:
        d = abs(s["t"] - t)
        if d <= max_dt and (best is None or d < abs(best["t"] - t)):
            best = s
    return best


def view(witness: dict, point: tuple[float, float], *, range_m: float,
         margin: float) -> str:
    """What the witness's sample can say about ``point``: ``out_of_range``,
    ``blocked`` (something nearer in that direction), or ``visible``."""
    wx, wy, wyaw = witness["pose"]
    dist = math.hypot(point[0] - wx, point[1] - wy)
    if dist > range_m:
        return "out_of_range"
    secs = witness.get("sectors") or []
    if not secs:
        return "blocked"
    n = len(secs)
    width = 2 * math.pi / n
    bearing = _wrap(math.atan2(point[1] - wy, point[0] - wx) - wyaw)
    k = int(((bearing + width / 2) % (2 * math.pi)) // width) % n
    # The subject's own body makes the nearest return about its radius
    # short of its centre; anything nearer than the margin is a blocker.
    return "visible" if secs[k] >= dist - margin else "blocked"


def check_claims(subject: list[dict], own: list[dict], *, range_m: float = 3.0,
                 match_m: float = 0.4, max_dt: float = 0.6,
                 margin: float = 0.35) -> list[dict[str, Any]]:
    """Per subject sample the witness could judge: whether the witness saw an
    object where the subject claims to be. Samples the witness cannot judge
    (no own sample near that time, out of range, view blocked) are left out."""
    out = []
    for s in subject:
        w = _nearest(own, s["t"], max_dt)
        if w is None:
            continue
        claimed = (s["pose"][0], s["pose"][1])
        if view(w, claimed, range_m=range_m, margin=margin) != "visible":
            continue
        seen = any(math.hypot(o[0] - claimed[0], o[1] - claimed[1]) <= match_m
                   for o in w.get("objects") or [])
        out.append({"t": s["t"], "claimed": list(claimed),
                    "witness_pose": w["pose"][:2], "seen": seen})
    return out


def conflict_runs(checks: list[dict], *, min_run: int = 3) -> list[list[dict]]:
    """Runs of at least ``min_run`` consecutive unseen claims (by time). One
    missed frame is noise; a run is a contradiction."""
    runs, cur = [], []
    for c in sorted(checks, key=lambda c: c["t"]):
        if not c["seen"]:
            cur.append(c)
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur = []
    if len(cur) >= min_run:
        runs.append(cur)
    return runs


def latest_position(samples: list[dict]) -> tuple[float, float] | None:
    if not samples:
        return None
    s = max(samples, key=lambda s: s["t"])
    return s["pose"][0], s["pose"][1]
