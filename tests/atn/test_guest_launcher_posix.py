"""Separate-uid guest tool runs (atn/guest_launcher.py), Linux + root only.

Runs where the launcher really runs: as root, e.g. inside the daemon image::

    docker run --rm --user 0 -v "$PWD:/src" -w /src -e PYTHONPATH=/src \\
        autonet-node:dev python tests/atn/test_guest_launcher_posix.py

(also collected by pytest; skipped unless Linux and euid 0). Each case uses
``spawn: true`` so the guest can run code that tool_guard does not see (a
plain ``cat`` / an unhooked python): what is pinned here is the OS boundary,
not the audit-hook tripwire.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import threading
import time

try:
    import pytest
except ImportError:          # standalone run inside the daemon image
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.skipif(
        not sys.platform.startswith("linux") or os.geteuid() != 0,
        reason="separate-uid launcher needs Linux and root")

GUEST_UID = 10002
POOL = list(range(GUEST_UID, GUEST_UID + 4))
DAEMON_UID = 10001


_KEEP: list = []        # keep the client end open (GC would close the fd)


def _start_launcher():
    from atn import guest_launcher, guest_sandbox
    master, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    launcher = guest_launcher._Launcher(POOL, "/tmp")
    threading.Thread(target=launcher.serve, args=(master,), daemon=True).start()
    os.environ["ATN_GUEST_LAUNCHER_FD"] = str(client.fileno())
    os.environ["ATN_GUEST_REQUIRE_UID"] = "1"
    os.environ["ATN_GUEST_UID"] = f"{POOL[0]}-{POOL[-1]}"
    guest_sandbox._CHANNEL = None
    _KEEP.append(client)
    return client


async def _run(code: str, args: dict | None = None, caps=None, timeout=30):
    from atn import guest_sandbox
    proc = await guest_sandbox.spawn_guest(code.encode(), caps or {},
                                           data_dir="/data/atn")
    assert proc.mode == "uid"
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(json.dumps(args or {}).encode()), timeout)
    finally:
        await proc.aclose()
    return proc.returncode, out.decode(), err.decode()


def _guest_procs() -> list[int]:
    pids = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                if os.stat(f"/proc/{name}").st_uid not in POOL:
                    continue
                with open(f"/proc/{name}/stat") as fh:
                    state = fh.read().rsplit(")", 1)[1].split()[0]
                if state != "Z":   # zombies: dead, awaiting the init reaper
                    pids.append(int(name))
            except OSError:
                pass
    return pids


def _plant() -> tuple[str, str]:
    d = tempfile.mkdtemp(prefix="atn-daemon-data-")
    path = os.path.join(d, "planted.txt")
    with open(path, "w") as fh:
        fh.write("PLANTED-UID-SECRET")
    os.chown(path, DAEMON_UID, DAEMON_UID)
    os.chmod(path, 0o600)
    os.chown(d, DAEMON_UID, DAEMON_UID)
    os.chmod(d, 0o700)
    return d, path


_SPAWN = (
    "import json, os, subprocess, sys\n"
    "a = json.load(sys.stdin)\n"
    "def sh(argv):\n"
    "    r = subprocess.run(argv, capture_output=True, text=True)\n"
    "    return [r.returncode, r.stdout, r.stderr[-200:]]\n"
    "print(json.dumps({'uid': os.getuid(), 'env': dict(os.environ),\n"
    "    'cat': sh(['cat', a['path']]),\n"
    "    'environ': sh(['cat', '/proc/%d/environ' % a['daemon_pid']]),\n"
    "    'kill': sh([sys.executable, '-c',\n"
    "                'import os, sys; os.kill(int(sys.argv[1]), 0)',\n"
    "                str(a['daemon_pid'])])}))\n"
)


def test_uid_isolation_file_env_signal():
    _start_launcher()
    os.environ["ATN_PLANTED_SECRET"] = "ENV-UID-SECRET"
    _, path = _plant()
    rc, out, err = asyncio.run(_run(
        _SPAWN, {"path": path, "daemon_pid": os.getpid()},
        caps={"spawn": True}))
    assert rc == 0, err
    res = json.loads(out)
    assert res["uid"] in POOL
    assert "PLANTED-UID-SECRET" not in out
    assert res["cat"][0] != 0                       # permission denied
    assert "ENV-UID-SECRET" not in out
    assert "ATN_PLANTED_SECRET" not in res["env"]
    assert res["environ"][0] != 0                   # daemon environ unreadable
    assert res["kill"][0] != 0                      # cannot signal the daemon


def test_uid_isolation_killed_on_timeout_including_escapees():
    _start_launcher()
    os.environ["ATN_GUEST_TOOL_TIMEOUT_S"] = "2"
    try:
        code = (
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c',\n"
            "    'import os, time; os.setsid(); time.sleep(300)'])\n"
            "time.sleep(300)\n"
        )
        t0 = time.monotonic()
        rc, out, err = asyncio.run(_run(code, caps={"spawn": True}))
        assert time.monotonic() - t0 < 20
        assert rc == 124 and "timed out" in err, (rc, err)
        time.sleep(0.5)
        assert _guest_procs() == []
    finally:
        os.environ.pop("ATN_GUEST_TOOL_TIMEOUT_S", None)


def test_concurrent_runs_get_distinct_uids():
    """Each live run has its own uid: runs cannot signal or read each other."""
    _start_launcher()
    code = ("import json, os, time\n"
            "time.sleep(1.5)\n"
            "print(json.dumps({'uid': os.getuid()}))\n")

    async def both():
        return await asyncio.gather(_run(code), _run(code))

    (rc1, out1, err1), (rc2, out2, err2) = asyncio.run(both())
    assert rc1 == 0 and rc2 == 0, (err1, err2)
    u1, u2 = json.loads(out1)["uid"], json.loads(out2)["uid"]
    assert u1 in POOL and u2 in POOL and u1 != u2


def test_escapee_with_scrubbed_environ_and_leftovers_cleared():
    """A child that drops every env marker and leaves the session is still
    killed (kill -1 as the run's uid), and files the run left in /tmp are
    gone before the uid is reused."""
    _start_launcher()
    left = f"/tmp/atn-leftover-{os.urandom(4).hex()}"
    code = (
        "import subprocess, sys\n"
        f"open({left!r}, 'w').write('x')\n"
        "subprocess.Popen([sys.executable, '-c',\n"
        "    'import os, time; os.setsid(); time.sleep(300)'], env={})\n"
        "print('{}')\n"
    )
    rc, out, err = asyncio.run(_run(code, caps={"spawn": True, "fs": True}))
    assert rc == 0, err
    time.sleep(0.5)
    assert _guest_procs() == []
    assert not os.path.exists(left)


def test_owner_listener_refuses_guest_uid():
    """A guest process outside tool_guard (unhooked python child) dials the
    real local owner listener and gets no owner session."""
    _start_launcher()
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime
    from atn.ws_server import WebSocketBridge

    port = 27741
    client = (
        "import asyncio, websockets\n"
        "async def main():\n"
        "    try:\n"
        f"        async with websockets.connect('ws://127.0.0.1:{port}') as ws:\n"
        "            print('FRAME ' + str(await asyncio.wait_for(ws.recv(), 5))[:30])\n"
        "    except websockets.ConnectionClosed as e:\n"
        "        print('CLOSED ' + str(e.rcvd.code if e.rcvd else None))\n"
        "asyncio.run(main())\n"
    )
    code = (
        "import json, subprocess, sys\n"
        f"r = subprocess.run([sys.executable, '-c', {client!r}],\n"
        "                   capture_output=True, text=True)\n"
        "print(json.dumps({'out': r.stdout.strip(), 'err': r.stderr[-300:]}))\n"
    )

    async def main():
        from pathlib import Path
        d = Path(tempfile.mkdtemp())
        cfg = ATNConfig(data_dir=d / "data", agents_dir=d / "agents")
        os.makedirs(cfg.data_dir, exist_ok=True)
        os.makedirs(cfg.agents_dir, exist_ok=True)
        cfg.autonet.enabled = False
        cfg.voice.enabled = False
        rt = Runtime(EventBus(), data_dir=cfg.data_dir, config=cfg)
        bridge = WebSocketBridge(rt, host="127.0.0.1", port=port)
        await bridge.start()
        try:
            rc, out, err = await _run(code, caps={"spawn": True, "net": True})
        finally:
            await bridge.stop()
        return rc, out, err

    rc, out, err = asyncio.run(main())
    assert rc == 0, err
    res = json.loads(out)
    assert res["out"].startswith("CLOSED 4403"), res


if __name__ == "__main__":
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        print("SKIP: needs Linux + root")
        sys.exit(0)
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
