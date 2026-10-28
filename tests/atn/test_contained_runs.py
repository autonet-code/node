"""Daemon-spawned runs never reach the owner listener, and authored tools get
no daemon secrets (docs_from_code/runtime_and_agents.md section 7, items 10
and 11).

Pinned here:
- an AUTHORED tool runs with the scrubbed environment (no daemon secret, only
  declared env names) and the keystore / data dir deny paths, including under
  ATN_AUTHORED_TOOLS_BARE=1 for the env;
- a connection whose client PID is a tracked tool / shell run, or a
  descendant of one, is refused by ws_auth.local_peer_denied; an untracked
  local client is still accepted;
- end to end against a live WebSocketBridge: an authored tool and an agent
  shell command that connect to the local listener get 4403, while the same
  client run by the test (an owner client) gets the snapshot.
"""
from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path

import pytest

from atn import contained_pids, shell_tools, ws_auth
from atn.agent_tools import execute_tool
from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.runtime import Runtime
from atn.ws_server import WebSocketBridge

_SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


def _runtime(tmp_path: Path) -> Runtime:
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


async def _authored(rt, code: str, capabilities: dict | None = None,
                    name: str = "own_tool"):
    await rt.register_agent(AgentDefinition(
        id="child", name="child", mode=AgentMode.COGNITIVE,
        system_prompt="c", cognitive_model="claude-sonnet-5"))
    args = {"name": name, "description": "d", "input_schema": _SCHEMA,
            "code": code}
    if capabilities is not None:
        args["capabilities"] = capabilities
    res = await execute_tool("register_tool", args, rt, caller_id="child")
    assert "error" not in res, res
    return rt.tool_store.get(res["digest"])


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    for pid in list(contained_pids.live()):
        contained_pids.unregister(pid)


# ---------------------------------------------------------------------------
# Authored tool environment and paths
# ---------------------------------------------------------------------------

_ENV_CODE = (
    "import sys, json, os\n"
    "json.load(sys.stdin)\n"
    "print(json.dumps({'secret': os.environ.get('ATN_TEST_SECRET'),"
    " 'owner': os.environ.get('BROKER_OWNER_SECRET'),"
    " 'declared': os.environ.get('ATN_TEST_DECLARED')}))\n"
)


@pytest.mark.asyncio
async def test_authored_tool_gets_no_daemon_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ATN_TEST_SECRET", "hunter2")
    monkeypatch.setenv("BROKER_OWNER_SECRET", "owner-secret")
    monkeypatch.setenv("ATN_TEST_DECLARED", "ok")
    rt = _runtime(tmp_path)
    rec = await _authored(rt, _ENV_CODE,
                          capabilities={"env": ["ATN_TEST_DECLARED"]})
    out = await rt.tool_store.call(rec, {}, caller_id="child")
    assert out["result"] == {"secret": None, "owner": None, "declared": "ok"}


@pytest.mark.asyncio
async def test_bare_mode_still_scrubs_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ATN_TEST_SECRET", "hunter2")
    monkeypatch.setenv("ATN_AUTHORED_TOOLS_BARE", "1")
    rt = _runtime(tmp_path)
    rec = await _authored(rt, _ENV_CODE)
    out = await rt.tool_store.call(rec, {}, caller_id="child")
    assert out["result"]["secret"] is None


@pytest.mark.asyncio
async def test_authored_tool_cannot_read_data_dir_or_keystore(tmp_path):
    rt = _runtime(tmp_path)
    planted = Path(rt._config.data_dir) / "private" / "planted.txt"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text("daemon secret")
    code = (
        "import sys, json\n"
        "json.load(sys.stdin)\n"
        f"print(open({str(planted)!r}).read())\n"
    )
    rec = await _authored(rt, code)
    out = await rt.tool_store.call(rec, {}, caller_id="child")
    assert "error" in out and "may not access" in out["error"], out

    argv, env, _ = rt.tool_store._exec_spec(rec, Path("x.py"))
    policy = json.loads(env["ATN_TOOL_POLICY"])
    from atn._vendor.kevin import keystore
    assert str(Path(keystore.KEYSTORE_DIR).resolve()) in policy["deny_paths"]


# ---------------------------------------------------------------------------
# Local listener PID gate
# ---------------------------------------------------------------------------

# Connects to argv[1] (port); with argv[2] == "spawn", does it from a child.
_CONNECTOR = (
    "import socket, subprocess, sys, time\n"
    "if len(sys.argv) > 2 and sys.argv[2] == 'spawn':\n"
    "    subprocess.run([sys.executable, sys.argv[0], sys.argv[1]])\n"
    "    sys.exit(0)\n"
    "s = socket.create_connection(('127.0.0.1', int(sys.argv[1])))\n"
    "time.sleep(30)\n"
)


async def _accept_from(tmp_path, *, track: str | None, via_child: bool):
    """Start a listener, run the connector (optionally tracked), and return
    local_peer_denied for the accepted connection."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    script = tmp_path / "connector.py"
    script.write_text(_CONNECTOR)
    argv = [sys.executable, str(script), str(port)]
    if via_child:
        argv.append("spawn")
    proc = await asyncio.create_subprocess_exec(*argv)
    if track:
        contained_pids.register(proc.pid, track)
    # A tracked run that is NOT the client keeps the gate live either way.
    decoy = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(30)")
    contained_pids.register(decoy.pid, "decoy tool")
    try:
        srv.settimeout(20)
        conn, peer = await asyncio.to_thread(srv.accept)
        try:
            return await asyncio.to_thread(
                ws_auth.local_peer_denied, peer, conn.getsockname())
        finally:
            conn.close()
    finally:
        for p in (proc, decoy):
            try:
                import psutil
                for c in psutil.Process(p.pid).children(recursive=True):
                    c.kill()
            except Exception:
                pass
            p.kill()
            await p.wait()
        srv.close()


@pytest.mark.asyncio
async def test_tracked_tool_pid_refused(tmp_path):
    reason = await _accept_from(tmp_path, track="authored tool", via_child=False)
    assert reason and "authored tool" in reason


@pytest.mark.asyncio
async def test_descendant_of_tracked_shell_refused(tmp_path):
    reason = await _accept_from(tmp_path, track="agent shell command",
                                via_child=True)
    assert reason and "agent shell command" in reason


@pytest.mark.asyncio
async def test_untracked_local_client_accepted(tmp_path):
    assert await _accept_from(tmp_path, track=None, via_child=False) is None


def test_gate_is_noop_without_tracked_runs():
    assert not contained_pids.live()
    assert ws_auth.local_peer_denied(("127.0.0.1", 5555), 7700) is None


def test_stale_entry_does_not_match_reused_pid(monkeypatch):
    import os
    contained_pids.register(os.getpid(), "tool")
    assert contained_pids.kind_of(os.getpid()) == "tool"
    # Same pid, different start time: a PID reused after a missed cleanup.
    contained_pids._STARTED[os.getpid()] -= 100.0
    assert contained_pids.kind_of(os.getpid()) is None


# ---------------------------------------------------------------------------
# End to end against the live listener
# ---------------------------------------------------------------------------

_WS_CLIENT = (
    "import asyncio, sys, websockets\n"
    "async def main():\n"
    "    try:\n"
    "        async with websockets.connect(sys.argv[1]) as ws:\n"
    "            msg = await asyncio.wait_for(ws.recv(), 5)\n"
    "            print('FRAME ' + str(msg)[:40])\n"
    "    except websockets.ConnectionClosed as e:\n"
    "        print('CLOSED ' + str(e.rcvd.code if e.rcvd else None))\n"
    "    except Exception as e:\n"
    "        print('ERR ' + type(e).__name__)\n"
    "asyncio.run(main())\n"
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_tool_and_shell_get_no_owner_session(tmp_path):
    rt = _runtime(tmp_path)
    port = _free_port()
    bridge = WebSocketBridge(rt, host="127.0.0.1", port=port)
    await bridge.start()
    client = tmp_path / "client.py"
    client.write_text(_WS_CLIENT)
    url = f"ws://127.0.0.1:{port}"
    try:
        # Owner client (not spawned by the daemon): served.
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(client), url, stdout=asyncio.subprocess.PIPE)
        out, _ = await asyncio.wait_for(proc.communicate(), 30)
        assert out.decode().startswith("FRAME"), out

        # Agent shell command running the same client: refused.
        res = await shell_tools.exec_bash(
            {"command": f'"{sys.executable}" "{client}" {url}', "timeout": 60})
        assert "CLOSED 4403" in res.get("output", ""), res

        # Authored tool that connects (in-process python, and via a child).
        code = (
            "import sys, json, subprocess\n"
            "json.load(sys.stdin)\n"
            f"r = subprocess.run([sys.executable, {str(client)!r}, {url!r}],"
            " capture_output=True, text=True)\n"
            "print(json.dumps({'out': r.stdout.strip(),"
            " 'err': r.stderr[-400:]}))\n"
        )
        # Declared env: where a user-site websockets install is found.
        rec = await _authored(rt, code, capabilities={
            "net": True, "fs": True, "spawn": True,
            "env": ["APPDATA", "HOME", "USERPROFILE"]})
        res = await rt.tool_store.call(rec, {}, caller_id="child")
        assert res["result"]["out"].startswith("CLOSED 4403"), res
        assert not contained_pids.live()
    finally:
        await bridge.stop()
