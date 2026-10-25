"""Safety pack runtime and worklog attestation (docs/robots.md)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from atn.robot import packs as P
from atn.robot import worklog as W

PACK_SRC = (Path(__file__).resolve().parents[2] / "robot" / "packs"
            / "proximity_stop.py").read_text("utf-8")


def _manifest(code: str, **extra) -> dict:
    return {"name": "proximity_stop", "trust_class": "pinned",
            "code_digest": hashlib.sha256(code.encode()).hexdigest(),
            "runtime": "python3", **extra}


def _loaded(tmp_path: Path, code: str = PACK_SRC) -> list[P.Pack]:
    P.write_bundle(tmp_path, [(_manifest(code), code)])
    res = P.load_bundle(tmp_path)
    assert res.ok, res.errors
    return res.packs


def _obs(front: float | None, now: float) -> dict:
    return {"stamp": now, "scan": {} if front is None else {"front_min": front}}


def _hb(now: float, ok: bool = True) -> dict:
    return {"ts": now, "sponsor_ok": ok}


def test_clear_path_passes(tmp_path):
    now = time.time()
    v = P.evaluate(_loaded(tmp_path), _obs(3.0, now),
                   {"linear": 0.3, "angular": 0.0}, now=now, heartbeat=_hb(now))
    assert v["verdict"] == P.PASS and v["cmd"]["linear"] == 0.3


def test_obstacle_stops_forward_but_allows_reverse(tmp_path):
    now = time.time()
    packs = _loaded(tmp_path)
    fwd = P.evaluate(packs, _obs(0.3, now), {"linear": 0.3, "angular": 0.0},
                     now=now, heartbeat=_hb(now))
    assert fwd["verdict"] == P.STOP and fwd["cmd"] == P.STOP_CMD
    back = P.evaluate(packs, _obs(0.3, now), {"linear": -0.2, "angular": 0.0},
                      now=now, heartbeat=_hb(now))
    assert back["verdict"] == P.PASS


def test_speed_is_capped_and_slowed_near_obstacles(tmp_path):
    now = time.time()
    packs = _loaded(tmp_path)
    fast = P.evaluate(packs, _obs(5.0, now), {"linear": 2.0, "angular": 0.0},
                      now=now, heartbeat=_hb(now))
    assert fast["verdict"] == P.CLIP and fast["cmd"]["linear"] == 0.5
    near = P.evaluate(packs, _obs(0.7, now), {"linear": 0.5, "angular": 0.0},
                      now=now, heartbeat=_hb(now))
    assert near["verdict"] == P.CLIP and 0 < near["cmd"]["linear"] < 0.5


@pytest.mark.parametrize("case,reason", [
    ("stale_obs", "sensor data stale"),
    ("stale_hb", "daemon heartbeat stale"),
    ("no_sponsor", "sponsor unreachable"),
    ("bad_cmd", "malformed command"),
])
def test_fail_closed_conditions(tmp_path, case, reason):
    now = time.time()
    obs = _obs(3.0, now - 5 if case == "stale_obs" else now)
    hb = _hb(now - 60 if case == "stale_hb" else now, ok=case != "no_sponsor")
    cmd = {"linear": "fast"} if case == "bad_cmd" else {"linear": 0.1, "angular": 0}
    v = P.evaluate(_loaded(tmp_path), obs, cmd, now=now, heartbeat=hb)
    assert v["verdict"] == P.STOP and v["reason"] == reason


def test_no_packs_stops():
    now = time.time()
    v = P.evaluate([], _obs(3.0, now), {"linear": 0.1, "angular": 0}, now=now)
    assert v["verdict"] == P.STOP


def test_tampered_code_refuses_to_load(tmp_path):
    P.write_bundle(tmp_path, [(_manifest(PACK_SRC), PACK_SRC)])
    code_file = next((tmp_path / "packs").glob("*.py"))
    code_file.write_text(PACK_SRC.replace("0.45", "0.01"), "utf-8")
    res = P.load_bundle(tmp_path)
    assert not res.ok and "code digest mismatch" in res.errors[0]


def test_tampered_manifest_refuses_to_load(tmp_path):
    [digest] = P.write_bundle(tmp_path, [(_manifest(PACK_SRC), PACK_SRC)])
    mf = tmp_path / "packs" / f"{digest}.json"
    m = json.loads(mf.read_text("utf-8"))
    m["name"] = "something else"
    mf.write_text(json.dumps(m, sort_keys=True), "utf-8")
    assert not P.load_bundle(tmp_path).ok


def test_pack_with_capabilities_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no capabilities"):
        P.write_bundle(tmp_path, [(_manifest(PACK_SRC, capabilities={"net": True}),
                                   PACK_SRC)])


def test_raising_pack_stops(tmp_path):
    code = "def check(obs, cmd):\n    raise RuntimeError('boom')\n"
    now = time.time()
    v = P.evaluate(_loaded(tmp_path, code), _obs(3.0, now),
                   {"linear": 0.1, "angular": 0}, now=now)
    assert v["verdict"] == P.STOP and "boom" in v["reason"]


def test_most_restrictive_pack_wins(tmp_path):
    clip = ("def check(obs, cmd):\n"
            "    return {'verdict': 'clip', 'cmd': {'linear': 0.1, 'angular': 0.0}}\n")
    stop = "def check(obs, cmd):\n    return {'verdict': 'stop', 'reason': 'no'}\n"
    P.write_bundle(tmp_path, [(_manifest(clip), clip), (_manifest(stop), stop)])
    packs = P.load_bundle(tmp_path).packs
    now = time.time()
    v = P.evaluate(packs, _obs(3.0, now), {"linear": 0.4, "angular": 0}, now=now)
    assert v["verdict"] == P.STOP and [p["verdict"] for p in v["packs"]] == ["clip", "stop"]


def test_pack_runs_as_a_tool(tmp_path):
    import subprocess
    import sys
    src = tmp_path / "p.py"
    src.write_text(PACK_SRC, "utf-8")
    out = subprocess.run([sys.executable, str(src)], capture_output=True, text=True,
                         input=json.dumps({"obs": {"scan": {"front_min": 0.2}},
                                           "cmd": {"linear": 0.3}}), check=True)
    assert json.loads(out.stdout)["verdict"] == "stop"


# --- worklog attestation ---------------------------------------------------

KEY = "0x" + "11" * 32


def _bags(tmp_path: Path, n: int = 3) -> Path:
    bag = tmp_path / "bag"
    bag.mkdir()
    now = time.time()
    for i in range(n):
        f = bag / f"worklog_{i}.mcap"
        f.write_bytes(os.urandom(64) + W.MCAP_MAGIC)
        os.utime(f, (now - (n - i), now - (n - i)))
    return bag


def _robot() -> str:
    from eth_account import Account
    return Account.from_key(KEY).address


# --- daemon side -------------------------------------------------------------

def _robot_runtime(tmp_path):
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime
    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir(parents=True)
    agents_dir.mkdir(parents=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.robot.bundle_dir = str(tmp_path / "bundle")
    config.robot.bag_dir = str(_bags(tmp_path))
    return Runtime(EventBus(), data_dir=data_dir, config=config), config


@pytest.mark.asyncio
async def test_service_exports_packs_and_attests(tmp_path):
    from atn.agent_tools import _create_agent
    from atn.robot.service import RobotService
    rt, config = _robot_runtime(tmp_path)
    reg = rt.tool_store.register(name="proximity_stop", description="stop pack",
                                 input_schema={"type": "object"}, author="user",
                                 code=PACK_SRC)
    out = await _create_agent(rt, {"id": "rover", "provider": "external"})
    config.robot.required_packs = ["proximity_stop"]
    config.robot.agent_id = "rover"
    svc = RobotService(rt, config)
    assert svc.export_packs() == {"ok": True, "packs": [reg["digest"]]}
    loaded = P.load_bundle(config.robot.bundle_dir)
    assert loaded.ok and loaded.packs[0].digest == reg["digest"]
    link = svc.attest("nonce-1")
    assert W.verify_link(link, robot=out["address"]) == []
    assert link["challenge"] == "nonce-1" and link["packs"] == [reg["digest"]]
    assert [lf["name"] for lf in link["leaves"]] == ["worklog_0.mcap", "worklog_1.mcap"]
    assert [c["challenge"] for c in P.read_challenges(config.robot.bundle_dir)] == ["nonce-1"]
    assert svc.attest("nonce-2")["prev"] != link["prev"]
    # Polls arriving together all reach the filter, oldest first.
    assert [c["challenge"] for c in P.read_challenges(config.robot.bundle_dir)] == \
        ["nonce-1", "nonce-2"]


@pytest.mark.asyncio
async def test_service_refuses_a_tool_with_capabilities(tmp_path):
    from atn.robot.service import RobotService
    rt, config = _robot_runtime(tmp_path)
    rt.tool_store.register(name="netpack", description="x",
                           input_schema={"type": "object"}, author="user",
                           code=PACK_SRC, capabilities={"net": True})
    config.robot.required_packs = ["netpack", "missing"]
    res = RobotService(rt, config).export_packs()
    assert not res["ok"] and len(res["errors"]) == 2
    assert not (Path(config.robot.bundle_dir) / "required.json").exists()


@pytest.mark.asyncio
async def test_registry_standards_switch_packs_and_fail_closed(tmp_path):
    from atn.agent_tools import _create_agent
    from atn.robot.service import RobotService
    rt, config = _robot_runtime(tmp_path)
    reg = rt.tool_store.register(name="proximity_stop", description="stop pack",
                                 input_schema={"type": "object"}, author="user",
                                 code=PACK_SRC)
    await _create_agent(rt, {"id": "rover", "provider": "external"})
    config.robot.required_packs = []                 # nothing from config
    config.robot.agent_id = "rover"
    svc = RobotService(rt, config)
    res = svc.apply_standards([reg["digest"]])
    assert res["ok"] and svc.exported == [reg["digest"]]
    assert svc.status()["required_from"] == "registry"
    assert P.load_bundle(config.robot.bundle_dir).ok
    assert svc.apply_standards([reg["digest"]]) is None          # unchanged
    # The jurisdiction adds a pack this daemon does not have: the robot stops.
    res = svc.apply_standards([reg["digest"], "e" * 64])
    assert not res["ok"] and svc.exported == []
    loaded = P.load_bundle(config.robot.bundle_dir)
    assert not loaded.ok and loaded.errors == ["no required packs"]
