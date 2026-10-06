"""Guest tool authoring over the integration listener (origin="integration").

docs/integration_listener.md, "Guest tool authoring": a token-bound guest may
register pinned code, and that record ALWAYS runs on the guest containment
path (atn/guest_sandbox.py). These tests run the same-uid fallback (the path a
dev box or Windows gets); the separate-uid launcher path is covered by
tests/atn/test_guest_launcher_posix.py (Linux, root, e.g. inside the daemon
image).

Pinned here: a guest tool cannot read a planted file in the data dir, cannot
see a planted env secret, cannot reach the owner socket as owner (both the
tool_guard tripwire and the listener's peer-credential gate), is killed on
timeout; the author can publish, a non-author cannot; a guest cannot forge
its origin or reach connectors/env/secrets.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time

import pytest
import websockets

from atn import guest_sandbox, ws_auth
from atn.config import ATNConfig
from atn.events import EventBus
from atn.models import AgentDefinition, AgentMode
from atn.runtime import Runtime
from atn.ws_server import (
    INTEGRATION_ALLOWED_MESSAGES,
    INTEGRATION_DENIED_MESSAGES,
    WebSocketBridge,
)

_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}}
_OWNER_WALLET = "0x" + "ab" * 20


@pytest.fixture(autouse=True)
def _same_uid_fallback(monkeypatch):
    # Force the fallback path regardless of the host env.
    monkeypatch.delenv("ATN_GUEST_LAUNCHER_FD", raising=False)
    monkeypatch.delenv("ATN_GUEST_REQUIRE_UID", raising=False)
    monkeypatch.delenv("ATN_GUEST_UID", raising=False)
    monkeypatch.setenv("ATN_GUEST_ALLOW_SAME_UID", "1")
    monkeypatch.setenv("ATN_GUEST_TOOL_TIMEOUT_S", "20")


async def _fleet(tmp_path, *, owner_wallet: str = ""):
    data_dir = tmp_path / "data"
    agents_dir = tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    config.autonet.owner_wallet = owner_wallet
    rt = Runtime(EventBus(), data_dir=data_dir, config=config)
    for aid, pid in (("root", None), ("guest", "root"), ("other", "root")):
        await rt.registry.register_agent(AgentDefinition(
            id=aid, name=aid, mode=AgentMode.COGNITIVE, parent_id=pid, budgets={}))
    return rt


def _bridge(rt, **kw):
    b = WebSocketBridge(rt, host="127.0.0.1", port=0, integration_burst=10_000, **kw)
    b._integration_store = ws_auth.IntegrationTokenStore(rt._config.data_dir)
    return b


def _session(rec):
    return ws_auth.IntegrationSession(token_id=rec.token_id,
                                      token_hash=rec.token_hash,
                                      agent_id=rec.agent_id, label=rec.label)


async def _register(bridge, sess, name, code, caps=None, **extra):
    msg = {"type": "register_tool", "msg_id": name, "name": name,
           "description": f"{name} tool", "input_schema": _SCHEMA, "code": code,
           **extra}
    if caps is not None:
        msg["capabilities"] = caps
    return await bridge._handle_integration_message(msg, sess)


async def _use(bridge, sess, digest, args=None):
    return await bridge._handle_integration_message(
        {"type": "use_tool", "msg_id": "u", "name": f"reg_{digest[:12]}",
         "arguments": args or {}}, sess)


_UPPER = (
    "import json, sys\n"
    "a = json.load(sys.stdin)\n"
    "print(json.dumps({'upper': a.get('text', '').upper()}))\n"
)


@pytest.mark.asyncio
async def test_register_tool_is_allowed_and_marks_origin(tmp_path):
    assert "register_tool" in INTEGRATION_ALLOWED_MESSAGES
    assert "register_tool" not in INTEGRATION_DENIED_MESSAGES
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    # A forged "_origin" is stripped with every other "_" key.
    resp = await _register(bridge, sess, "upper_guest", _UPPER,
                           _origin="authored")
    assert resp["ok"] is True, resp
    digest = resp["result"]["digest"]
    record = rt.tool_store.get(digest)
    assert record.origin == "integration"
    assert record.author_id == "guest"
    assert record.published is False
    # Runs and returns a result on the guest path.
    out = await _use(bridge, sess, digest, {"text": "hello"})
    assert out["ok"] is True, out
    assert out["result"]["result"] == {"upper": "HELLO"}
    # The flag survives a reload of the store.
    from atn.tool_store import ToolStore
    reloaded = ToolStore(rt, rt.tool_store._dir)
    assert reloaded.get(digest).origin == "integration"
    # list_tools marks it as the guest's own.
    lt = await bridge._handle_integration_message(
        {"type": "list_tools", "msg_id": "l"}, sess)
    row = next(t for t in lt["result"]["tools"] if t["digest"] == digest)
    assert row["mine"] is True and row["origin"] == "integration"


@pytest.mark.asyncio
async def test_register_refuses_connectors_env_secrets(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    r = await _register(bridge, sess, "conn_c1", "", connector_id="gmail")
    assert r["ok"] is False and r["code"] == "not_allowed", r
    r = await _register(bridge, sess, "conn_c2", "print(1)", provider="google")
    assert r["ok"] is False and r["code"] == "not_allowed", r
    r = await _register(bridge, sess, "conn_c3", "")
    assert r["ok"] is False and r["code"] == "bad_request", r
    for caps in ({"env": ["HOME"]}, {"secrets": ["openai"]},
                 {"provides": ["bash"]}, {"net": "yes"}):
        r = await _register(bridge, sess, "conn_c4", "print(1)", caps)
        assert r["ok"] is False and r["code"] == "bad_request", (caps, r)
    # A composite reaching a connector-backed tool is refused.
    conn = rt.tool_store.register(
        name="gmail_send", description="x", input_schema=_SCHEMA,
        author="guest", connector_id="gmail")
    r = await _register(bridge, sess, "conn_c5", "print(1)",
                        dependencies=[conn["digest"]])
    assert r["ok"] is False and r["code"] == "not_allowed", r
    # The store enforces the same rules for any caller of origin=integration.
    with pytest.raises(ValueError):
        rt.tool_store.register(name="x", description="x", input_schema=_SCHEMA,
                               author="guest", code="print(1)",
                               capabilities={"env": ["PATH"]},
                               origin="integration")


@pytest.mark.asyncio
async def test_guest_tool_cannot_read_planted_data_file(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    planted = rt._config.data_dir / "keystore" / "planted_secret.txt"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text("PLANTED-DATA-SECRET-123")
    code = (
        "import json, os, sys\n"
        "p = json.load(sys.stdin)['path']\n"
        "out = {}\n"
        "try:\n"
        "    out['read'] = open(p).read()\n"
        "except Exception as e:\n"
        "    out['err'] = type(e).__name__\n"
        "try:\n"
        "    out['listing'] = os.listdir(os.path.dirname(p))\n"
        "except Exception as e:\n"
        "    out['list_err'] = type(e).__name__\n"
        "out['cwd'] = os.getcwd()\n"
        "print(json.dumps(out))\n"
    )
    for caps in (None, {"fs": True}):
        r = await _register(bridge, sess, f"reader_{str(bool(caps)).lower()}", code, caps)
        assert r["ok"], r
        out = await _use(bridge, sess, r["result"]["digest"], {"path": str(planted)})
        assert out["ok"] is True, out
        res = out["result"]["result"]
        assert "PLANTED-DATA-SECRET-123" not in json.dumps(res), (caps, res)
        assert res.get("err") == "PermissionError", res
        # Sandbox cwd is outside the data dir.
        assert not os.path.realpath(res["cwd"]).startswith(
            os.path.realpath(rt._config.data_dir))
        if caps:
            assert "planted_secret.txt" not in json.dumps(res.get("listing", []))


@pytest.mark.asyncio
async def test_guest_tool_cannot_see_env_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("ATN_PLANTED_SECRET", "ENV-SECRET-456")
    monkeypatch.setenv("AUTONET_RPC_URL", "https://rpc.example/key=ENV-SECRET-789")
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    code = "import json, os\nprint(json.dumps(dict(os.environ)))\n"
    r = await _register(bridge, sess, "env_dump", code, {"net": True, "fs": True})
    assert r["ok"], r
    out = await _use(bridge, sess, r["result"]["digest"])
    assert out["ok"] is True, out
    env = out["result"]["result"]
    blob = json.dumps(env)
    assert "ENV-SECRET" not in blob
    assert "ATN_PLANTED_SECRET" not in env and "AUTONET_RPC_URL" not in env
    assert "KEYSTORE_DIR" not in env and "ATN_TOOL_POLICY" not in env  # popped


@pytest.mark.asyncio
async def test_guest_tool_cannot_reach_owner_socket(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = WebSocketBridge(rt, host="127.0.0.1", port=27731,
                             integration_burst=10_000)
    bridge._integration_store = ws_auth.IntegrationTokenStore(rt._config.data_dir)
    await bridge.start()
    try:
        _, rec = bridge._integration_store.create("guest", "g")
        sess = _session(rec)
        code = (
            "import json, socket\n"
            "out = {}\n"
            "for host in ('127.0.0.1', 'localhost', '::1', '0.0.0.0'):\n"
            "    s = socket.socket(socket.AF_INET6 if ':' in host else socket.AF_INET)\n"
            "    s.settimeout(1)\n"
            "    try:\n"
            "        s.connect((host, 27731))\n"
            "        out[host] = 'CONNECTED'\n"
            "    except PermissionError:\n"
            "        out[host] = 'BLOCKED'\n"
            "    except Exception as e:\n"
            "        out[host] = type(e).__name__\n"
            "print(json.dumps(out))\n"
        )
        r = await _register(bridge, sess, "dial_owner", code, {"net": True})
        assert r["ok"], r
        out = await _use(bridge, sess, r["result"]["digest"])
        assert out["ok"] is True, out
        res = out["result"]["result"]
        assert "CONNECTED" not in res.values(), res
        assert res["127.0.0.1"] == "BLOCKED" and res["localhost"] == "BLOCKED"
    finally:
        await bridge.stop()


_WS_CLIENT = (
    "import asyncio, sys, websockets\n"
    "async def main():\n"
    "    sys.stdin.readline()\n"
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


@pytest.mark.asyncio
async def test_owner_listener_refuses_guest_process_by_peer_credential(tmp_path):
    """The OS-level gate behind the tripwire: a process the daemon tracks as a
    guest tool (here an UNguarded python, i.e. what native code that slipped
    past tool_guard would be) connects to the local listener and gets no
    owner session; the same client untracked gets the snapshot."""
    rt = await _fleet(tmp_path)
    port = 27732
    bridge = WebSocketBridge(rt, host="127.0.0.1", port=port)
    await bridge.start()
    script = tmp_path / "client.py"
    script.write_text(_WS_CLIENT)

    async def _run(as_guest: bool) -> str:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script), f"ws://127.0.0.1:{port}",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        if as_guest:
            guest_sandbox.register_guest_pid(proc.pid)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(b"go\n"), 30)
        finally:
            guest_sandbox.unregister_guest_pid(proc.pid)
        return out.decode().strip()

    try:
        assert (await _run(False)).startswith("FRAME {\"type\": \"snapshot\"")
        refused = await _run(True)
        assert refused.startswith("CLOSED 4403"), refused
        # And the gate is a no-op again once no guest run is live.
        assert (await _run(False)).startswith("FRAME")
    finally:
        await bridge.stop()


def test_local_peer_denied_uid_gate(monkeypatch):
    monkeypatch.setenv("ATN_GUEST_UID", "10002")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ws_auth, "_linux_peer_uid", lambda p, s: 10002)
    assert "guest tool uid" in ws_auth.local_peer_denied(("127.0.0.1", 5555), 7700)
    monkeypatch.setattr(ws_auth, "_linux_peer_uid", lambda p, s: 10001)
    assert ws_auth.local_peer_denied(("127.0.0.1", 5555), 7700) is None
    # Unresolvable peer while uid isolation is on: fail closed.
    monkeypatch.setattr(ws_auth, "_linux_peer_uid", lambda p, s: None)
    assert ws_auth.local_peer_denied(("127.0.0.1", 5555), 7700)
    monkeypatch.delenv("ATN_GUEST_UID")
    assert ws_auth.local_peer_denied(("127.0.0.1", 5555), 7700) is None


@pytest.mark.asyncio
async def test_guest_tool_killed_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("ATN_GUEST_TOOL_TIMEOUT_S", "2")
    seen: list[int] = []
    real = guest_sandbox.register_guest_pid
    monkeypatch.setattr(guest_sandbox, "register_guest_pid",
                        lambda pid: (seen.append(pid), real(pid)))
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    r = await _register(bridge, sess, "sleeper",
                        "import time\nwhile True:\n    time.sleep(0.1)\n")
    assert r["ok"], r
    t0 = time.monotonic()
    out = await _use(bridge, sess, r["result"]["digest"])
    assert time.monotonic() - t0 < 15
    assert out["ok"] is False and "timed out" in out["error"], out
    import psutil
    assert seen
    for pid in seen:
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == "zombie"
    assert not guest_sandbox.live_guest_pids()


@pytest.mark.asyncio
async def test_guest_composite_runs_contained(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    dep = await _register(bridge, sess, "upper_dep", _UPPER)
    assert dep["ok"], dep
    dep_digest = dep["result"]["digest"]
    code = (
        "import json, os, sys\n"
        "a = json.loads(sys.stdin.readline())\n"
        f"print(json.dumps({{'call': '{dep_digest}', 'args': a}}), flush=True)\n"
        "r = json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'return': {'dep': r, 'secret': os.environ.get('ATN_PLANTED_SECRET')}}), flush=True)\n"
    )
    os.environ["ATN_PLANTED_SECRET"] = "COMPOSITE-SECRET"
    try:
        comp = await _register(bridge, sess, "comp", code,
                               dependencies=[dep_digest])
        assert comp["ok"], comp
        out = await _use(bridge, sess, comp["result"]["digest"], {"text": "ab"})
    finally:
        os.environ.pop("ATN_PLANTED_SECRET", None)
    assert out["ok"] is True, out
    res = out["result"]["result"]
    assert res["dep"]["result"] == {"upper": "AB"}
    assert res["secret"] is None


@pytest.mark.asyncio
async def test_require_uid_refuses_without_launcher(tmp_path, monkeypatch):
    monkeypatch.setenv("ATN_GUEST_REQUIRE_UID", "1")
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    r = await _register(bridge, sess, "upper_req", _UPPER)
    assert r["ok"], r
    out = await _use(bridge, sess, r["result"]["digest"], {"text": "x"})
    assert out["ok"] is False and "uid isolation" in out["error"], out
    # Even the owner calling it in-daemon gets no weaker path.
    record = rt.tool_store.get(r["result"]["digest"])
    direct = await rt.tool_store.call(record, {"text": "x"}, caller_id=None)
    assert "uid isolation" in direct.get("error", ""), direct


@pytest.mark.asyncio
async def test_publish_author_only(tmp_path):
    rt = await _fleet(tmp_path, owner_wallet=_OWNER_WALLET)
    bridge = _bridge(rt)
    _, grec = bridge._integration_store.create("guest", "g")
    _, orec = bridge._integration_store.create("other", "o")
    gsess, osess = _session(grec), _session(orec)
    r = await _register(bridge, gsess, "upper_pub", _UPPER)
    assert r["ok"], r
    digest = r["result"]["digest"]
    # Non-author (another guest token) cannot publish it.
    no = await bridge._handle_integration_message(
        {"type": "publish_tool", "msg_id": "p1", "digest": digest}, osess)
    assert no["ok"] is False, no
    assert rt.tool_store.get(digest).published is False
    # The author can.
    yes = await bridge._handle_integration_message(
        {"type": "publish_tool", "msg_id": "p2", "digest": digest}, gsess)
    assert yes["ok"] is True, yes
    record = rt.tool_store.get(digest)
    assert record.published is True and record.origin == "integration"
    # Provenance stamp, signed with the manifest: adopters see a guest wrote it.
    assert record.manifest.get("authored_via") == "integration"
    # Consensus author is the claimable owner wallet (unregistered agent).
    assert record.author == _OWNER_WALLET


@pytest.mark.asyncio
async def test_same_uid_fallback_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ATN_GUEST_ALLOW_SAME_UID", raising=False)
    assert guest_sandbox.containment_mode() == "refused"
    assert "refused" in (guest_sandbox.boot_warning() or "")
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    r = await _register(bridge, sess, "upper_off", _UPPER)
    assert r["ok"], r
    out = await _use(bridge, sess, r["result"]["digest"], {"text": "x"})
    assert out["ok"] is False and "ATN_GUEST_ALLOW_SAME_UID" in out["error"], out
    monkeypatch.setenv("ATN_GUEST_ALLOW_SAME_UID", "1")
    assert guest_sandbox.containment_mode() == "same-uid"
    assert "DAEMON user" in guest_sandbox.boot_warning()


@pytest.mark.asyncio
async def test_guest_composite_deps_run_as_author_not_caller(tmp_path):
    """The owner calling a guest composite does not lend the composite its
    authority: a dep the guest author may not call is refused."""
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    owned = rt.tool_store.register(
        name="owner_upper", description="owner tool", input_schema=_SCHEMA,
        author="root", code=_UPPER)
    od = owned["digest"]
    rt.tool_store.grant(od, "guest")
    code = (
        "import json, sys\n"
        "a = json.loads(sys.stdin.readline())\n"
        f"print(json.dumps({{'call': '{od}', 'args': a}}), flush=True)\n"
        "r = json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'return': r}), flush=True)\n"
    )
    comp = await _register(bridge, sess, "comp_owner", code, dependencies=[od])
    assert comp["ok"], comp
    record = rt.tool_store.get(comp["result"]["digest"])
    # Granted: the owner's call runs the dep as the guest author.
    ok = await rt.tool_store.call(record, {"text": "ab"}, caller_id=None)
    assert ok["result"] == {"result": {"upper": "AB"}}, ok
    # Grant revoked: the owner may call the dep, the guest author may not.
    rt.tool_store.revoke(od, "guest")
    no = await rt.tool_store.call(record, {"text": "ab"}, caller_id=None)
    assert "composite author not authorized" in json.dumps(no), no


@pytest.mark.asyncio
async def test_guest_may_not_depend_on_secret_bound_tool(tmp_path):
    rt = await _fleet(tmp_path)
    owned = rt.tool_store.register(
        name="owner_secret", description="uses a key", input_schema=_SCHEMA,
        author="root", code=_UPPER, capabilities={"secrets": ["openai"]})
    with pytest.raises(ValueError, match="secrets or env"):
        rt.tool_store.register(
            name="guest_wrap", description="wraps", input_schema=_SCHEMA,
            author="guest", code=_UPPER, dependencies=[owned["digest"]],
            origin="integration")


@pytest.mark.asyncio
async def test_guest_register_name_rules(tmp_path):
    rt = await _fleet(tmp_path)
    bridge = _bridge(rt)
    _, rec = bridge._integration_store.create("guest", "g")
    sess = _session(rec)
    bad = await _register(bridge, sess, "atn_shadow", _UPPER)
    assert bad["ok"] is False and "may not start" in bad["error"], bad
    bad = await _register(bridge, sess, "Bad-Name", _UPPER)
    assert bad["ok"] is False and "snake_case" in bad["error"], bad
    rt.tool_store.register(name="owner_named", description="owner",
                           input_schema=_SCHEMA, author="root", code=_UPPER)
    taken = await _register(bridge, sess, "owner_named", _UPPER)
    assert taken["ok"] is False and "taken" in taken["error"], taken
    assert rt.tool_store.resolve("owner_named") is not None


def _proc_row(local: str, remote: str, uid: int) -> str:
    return f"   0: {local} {remote} 01 00000000:00000000 00:00000000 00000000 {uid} 0 1 1\n"


def test_linux_peer_uid_matches_full_four_tuple(monkeypatch):
    """The daemon's accepted socket must never be taken for the client end,
    even when the client's source port equals the server port."""
    import io

    def fake_tables(rows):
        def _open(path, *a, **kw):
            if path == "/proc/net/tcp":
                return io.StringIO("  sl local rem st\n" + "".join(rows))
            raise OSError(path)
        return _open

    srv = ("127.0.0.1", 7700)
    # 127.0.0.1 = 0100007F, 127.0.0.2 = 0200007F, 7700 = 1E14, 5555 = 15B3.
    accepted = _proc_row("0100007F:1E14", "0200007F:1E14", 10001)
    client = _proc_row("0200007F:1E14", "0100007F:1E14", 10002)
    monkeypatch.setattr(ws_auth, "open", fake_tables([accepted, client]),
                        raising=False)
    # Source port == server port: unresolvable, so the caller refuses.
    assert ws_auth._linux_peer_uid(("127.0.0.2", 7700), srv) is None
    normal = [_proc_row("0100007F:1E14", "0100007F:15B3", 10001),
              _proc_row("0100007F:15B3", "0100007F:1E14", 10002)]
    monkeypatch.setattr(ws_auth, "open", fake_tables(normal), raising=False)
    assert ws_auth._linux_peer_uid(("127.0.0.1", 5555), srv) == 10002
    # Same ports, different peer address: no match.
    assert ws_auth._linux_peer_uid(("127.0.0.3", 5555), srv) is None
    # Two rows for one 4-tuple: ambiguous, refused.
    dup = normal + [_proc_row("0100007F:15B3", "0100007F:1E14", 10001)]
    monkeypatch.setattr(ws_auth, "open", fake_tables(dup), raising=False)
    assert ws_auth._linux_peer_uid(("127.0.0.1", 5555), srv) is None
    # And through the gate: unresolvable with uid isolation on => refused.
    monkeypatch.setenv("ATN_GUEST_UID", "10002-10017")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ws_auth, "open", fake_tables([accepted, client]),
                        raising=False)
    assert ws_auth.local_peer_denied(("127.0.0.2", 7700), srv)


def test_guest_uids_parses_pool_range(monkeypatch):
    monkeypatch.setenv("ATN_GUEST_UID", "10002-10004,20000")
    assert guest_sandbox.guest_uids() == {10002, 10003, 10004, 20000}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_tool_guard_blocks_fork_without_spawn(tmp_path):
    script = tmp_path / "t.py"
    script.write_text("import os\nos.fork()\nprint('forked')\n")
    guard = os.path.join(os.path.dirname(guest_sandbox.__file__), "tool_guard.py")
    env = dict(os.environ, ATN_TOOL_POLICY=json.dumps({"spawn": False}))
    r = subprocess.run([sys.executable, guard, str(script)], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "undeclared capability: spawn (os.fork)" in r.stderr
