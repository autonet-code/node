"""Safety pack runtime (docs/robots.md, "Pack contract").

Stdlib only, on purpose: the same file is loaded by the daemon (to test and
export packs) and by the ROS 2 filter node on the robot, which has none of the
daemon's dependencies.

A pack is a pinned tool whose code defines ``check(obs, cmd) -> verdict``. The
robot bundle on disk is one directory:

    required.json            {"packs": [<manifest digest>, ...]}
    heartbeat.json           {"ts": <unix seconds>, "sponsor_ok": bool}
    packs/<digest>.json      the signed manifest, as stored in the blob store
    packs/<code_digest>.py   the pack's code

Every failure is a stop: a missing or tampered pack, a pack that raises, a bad
verdict, stale sensors, a stale heartbeat or no sponsor.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PASS, CLIP, STOP = "pass", "clip", "stop"
_RANK = {PASS: 0, CLIP: 1, STOP: 2}

STOP_CMD = {"linear": 0.0, "angular": 0.0}

# Defaults the filter node can override.
OBS_MAX_AGE_S = 0.5
HEARTBEAT_MAX_AGE_S = 5.0
# A pack runs in-process and cannot be preempted, so an over-budget check is
# caught after the fact: its command is still never forwarded.
PACK_BUDGET_S = 0.01


def manifest_digest(manifest: dict[str, Any]) -> str:
    """The blob store's digest of a manifest (BlobStore.add_json)."""
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


def _stop(reason: str) -> dict[str, Any]:
    return {"verdict": STOP, "cmd": dict(STOP_CMD), "reason": reason}


@dataclass
class Pack:
    digest: str
    name: str
    check: Callable[[dict, dict], dict]


@dataclass
class LoadResult:
    packs: list[Pack] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.packs)


def compile_pack(code: str, label: str = "pack") -> Callable[[dict, dict], dict]:
    """Execute pack source and return its ``check``. Raises ValueError."""
    namespace: dict[str, Any] = {"__name__": f"atn_pack_{label}"}
    exec(compile(code, f"<pack {label}>", "exec"), namespace)  # noqa: S102
    check = namespace.get("check")
    if not callable(check):
        raise ValueError("pack code defines no check(obs, cmd)")
    return check


def pack_errors(manifest: dict[str, Any]) -> list[str]:
    """Why a manifest cannot be a safety pack (empty list = it can)."""
    errors = []
    if manifest.get("trust_class") != "pinned" or not manifest.get("code_digest"):
        errors.append("a pack must be a pinned tool with code")
    if manifest.get("capabilities"):
        errors.append("a pack must declare no capabilities")
    if manifest.get("dependencies"):
        errors.append("a pack must not depend on other tools")
    return errors


def load_bundle(bundle_dir: str | Path) -> LoadResult:
    """Load and verify every required pack. Any error leaves ``ok`` False."""
    root = Path(bundle_dir)
    out = LoadResult()
    try:
        required = json.loads((root / "required.json").read_text("utf-8"))
        digests = [str(d) for d in required.get("packs") or []]
    except Exception as exc:  # noqa: BLE001
        out.errors.append(f"required.json unreadable: {exc}")
        return out
    if not digests:
        out.errors.append("no required packs")
        return out
    for digest in digests:
        try:
            manifest = json.loads(
                (root / "packs" / f"{digest}.json").read_text("utf-8"))
            if manifest_digest(manifest) != digest:
                raise ValueError("manifest digest mismatch")
            errs = pack_errors(manifest)
            if errs:
                raise ValueError("; ".join(errs))
            code_bytes = (root / "packs" / f"{manifest['code_digest']}.py"
                          ).read_bytes()
            if hashlib.sha256(code_bytes).hexdigest() != manifest["code_digest"]:
                raise ValueError("code digest mismatch")
            check = compile_pack(code_bytes.decode("utf-8"), digest[:12])
            out.packs.append(Pack(digest=digest,
                                  name=str(manifest.get("name") or ""),
                                  check=check))
        except Exception as exc:  # noqa: BLE001
            out.errors.append(f"pack {digest[:12]}: {exc}")
    return out


def read_heartbeat(bundle_dir: str | Path) -> dict[str, Any]:
    try:
        return json.loads((Path(bundle_dir) / "heartbeat.json").read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _valid_cmd(cmd: Any) -> bool:
    """Every field a finite number. NaN slips past every comparison a pack
    makes (``NaN > 0`` is False), so a NaN speed must be malformed, not
    motion."""
    return (isinstance(cmd, dict)
            and all(isinstance(cmd.get(k), (int, float))
                    and not isinstance(cmd.get(k), bool)
                    and math.isfinite(cmd.get(k)) for k in STOP_CMD))


def _fresh(stamp: Any, now: float, max_age_s: float) -> bool:
    """Written as "not older than", so a NaN or missing stamp is stale."""
    try:
        age = now - float(stamp or 0)
    except (TypeError, ValueError):
        return False
    return age <= max_age_s


def evaluate(packs: list[Pack], obs: dict[str, Any], cmd: dict[str, Any], *,
             now: float | None = None,
             heartbeat: dict[str, Any] | None = None,
             obs_max_age_s: float = OBS_MAX_AGE_S,
             heartbeat_max_age_s: float = HEARTBEAT_MAX_AGE_S,
             pack_budget_s: float = PACK_BUDGET_S,
             ) -> dict[str, Any]:
    """Run every pack over one proposed command. Returns the final verdict
    plus ``packs`` (per-pack verdicts) for the event log."""
    now = time.time() if now is None else now
    if not packs:
        return {**_stop("no packs loaded"), "packs": []}
    if heartbeat is not None:
        if not _fresh(heartbeat.get("ts"), now, heartbeat_max_age_s):
            return {**_stop("daemon heartbeat stale"), "packs": []}
        if not heartbeat.get("sponsor_ok"):
            return {**_stop("sponsor unreachable"), "packs": []}
    if not _fresh(obs.get("stamp"), now, obs_max_age_s):
        return {**_stop("sensor data stale"), "packs": []}
    if not _valid_cmd(cmd):
        return {**_stop("malformed command"), "packs": []}

    current = {k: float(cmd[k]) for k in STOP_CMD}
    final, reasons, trail = PASS, [], []
    for pack in packs:
        try:
            t0 = time.perf_counter()
            v = pack.check(dict(obs), dict(current))
            if time.perf_counter() - t0 > pack_budget_s:
                raise ValueError("over time budget")
            verdict = v.get("verdict")
            if verdict not in _RANK:
                raise ValueError(f"bad verdict {verdict!r}")
            if verdict == CLIP:
                if not _valid_cmd(v.get("cmd")):
                    raise ValueError("clip without a valid cmd")
                current = {k: float(v["cmd"][k]) for k in STOP_CMD}
        except Exception as exc:  # noqa: BLE001
            v, verdict = {"reason": f"pack error: {exc}"}, STOP
        trail.append({"pack": pack.digest, "name": pack.name,
                      "verdict": verdict, "reason": str(v.get("reason") or "")})
        if verdict != PASS:
            reasons.append(f"{pack.name}: {v.get('reason') or verdict}")
        if _RANK[verdict] > _RANK[final]:
            final = verdict
        if verdict == STOP:
            break
    out_cmd = dict(STOP_CMD) if final == STOP else current
    return {"verdict": final, "cmd": out_cmd, "reason": "; ".join(reasons),
            "packs": trail}


def _replace(tmp: Path, dest: Path, attempts: int = 20) -> None:
    """Atomic rename, retried: on Windows it fails while another process
    (the robot's filter, through a bind mount) has the target open."""
    for i in range(attempts):
        try:
            tmp.replace(dest)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.01)


def write_heartbeat(bundle_dir: str | Path, *, sponsor_ok: bool,
                    now: float | None = None) -> None:
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "heartbeat.json.tmp"
    tmp.write_text(json.dumps({"ts": time.time() if now is None else now,
                               "sponsor_ok": bool(sponsor_ok)}), "utf-8")
    _replace(tmp, root / "heartbeat.json")


CHALLENGE_QUEUE = 32


def write_challenge(bundle_dir: str | Path, challenge: str,
                    now: float | None = None) -> None:
    """Hand a poll's challenge to the filter node, which publishes it on
    ``autonet/challenge`` so the next worklog chunk records it. The file
    keeps the last few challenges, so polls arriving together all land."""
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    queue = read_challenges(root)[-(CHALLENGE_QUEUE - 1):]
    queue.append({"ts": time.time() if now is None else now, "challenge": challenge})
    tmp = root / "challenge.json.tmp"
    tmp.write_text(json.dumps({"challenges": queue}), "utf-8")
    _replace(tmp, root / "challenge.json")


def read_challenges(bundle_dir: str | Path) -> list[dict[str, Any]]:
    try:
        data = json.loads((Path(bundle_dir) / "challenge.json").read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return []
    queue = data.get("challenges") if isinstance(data, dict) else None
    return [c for c in queue or [] if isinstance(c, dict) and c.get("challenge")]


def clear_bundle(bundle_dir: str | Path) -> None:
    """Require nothing loadable: the filter then refuses to move."""
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "required.json.tmp"
    tmp.write_text(json.dumps({"packs": []}), "utf-8")
    _replace(tmp, root / "required.json")


def write_bundle(bundle_dir: str | Path,
                 packs: list[tuple[dict[str, Any], str]]) -> list[str]:
    """Export (manifest, code) pairs as the required set. Returns digests.
    Refuses any manifest that cannot be a pack."""
    root = Path(bundle_dir)
    (root / "packs").mkdir(parents=True, exist_ok=True)
    digests = []
    for manifest, code in packs:
        errs = pack_errors(manifest)
        if errs:
            raise ValueError(f"{manifest.get('name')!r}: {'; '.join(errs)}")
        if hashlib.sha256(code.encode("utf-8")).hexdigest() != manifest["code_digest"]:
            raise ValueError(f"{manifest.get('name')!r}: code does not match manifest")
        digest = manifest_digest(manifest)
        (root / "packs" / f"{digest}.json").write_text(
            json.dumps(manifest, sort_keys=True), "utf-8")
        (root / "packs" / f"{manifest['code_digest']}.py").write_bytes(
            code.encode("utf-8"))
        digests.append(digest)
    tmp = root / "required.json.tmp"
    tmp.write_text(json.dumps({"packs": digests}), "utf-8")
    _replace(tmp, root / "required.json")
    return digests
