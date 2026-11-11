"""The service runner (atn/service_runner.py): the daemon runs the tools
behind session services itself. A real session tool (a TCP echo server,
stdlib only) registered in a real ToolStore: one shared process restarted
after a crash with the error recorded, a process per session, the session
cap, and the tool's usage reports reaching the session that watches them.

Run: python -m pytest tests/atn/test_service_runner.py -v
"""
from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path

import pytest

from atn import service_runner
from atn.service_runner import MAX_REPORT_BYTES, Lease, ServiceRunner, runner_for

# The tool: serves on AUTONET_SERVICE_PORT, echoes, and after each reply
# reports the bytes echoed as output tokens for that connection (its remote
# port, as the tool sees it). "whoami" answers its pid and settings;
# "crash" kills it.
TOOL = r'''
import http.client, json, os, socket, threading
from urllib.parse import urlparse

PORT = int(os.environ["AUTONET_SERVICE_PORT"])
USAGE = urlparse(os.environ.get("AUTONET_USAGE_URL", ""))
SETTINGS = json.loads(os.environ.get("AUTONET_SERVICE_SETTINGS", "{}"))

def report(port, n):
    body = json.dumps({"connection": port, "output_tokens": n})
    try:
        c = http.client.HTTPConnection(USAGE.hostname, USAGE.port, timeout=5)
        c.request("POST", USAGE.path, body, {"Content-Type": "application/json"})
        c.getresponse().read()
        c.close()
    except Exception:
        pass

def serve(conn, addr):
    with conn:
        while True:
            data = conn.recv(65536)
            if not data:
                return
            if data == b"crash":
                os._exit(3)
            if data == b"whoami":
                conn.sendall(json.dumps({"pid": os.getpid(), "settings": SETTINGS}).encode())
                continue
            conn.sendall(data)
            report(addr[1], len(data))

srv = socket.socket()
srv.bind(("127.0.0.1", PORT))
srv.listen(16)
while True:
    conn, addr = srv.accept()
    threading.Thread(target=serve, args=(conn, addr), daemon=True).start()
'''

SCHEMA = {"type": "object"}


def _make_runtime(tmp_path: Path):
    from atn.config import ATNConfig
    from atn.events import EventBus
    from atn.runtime import Runtime

    data_dir, agents_dir = tmp_path / "data", tmp_path / "agents"
    data_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    config = ATNConfig(data_dir=data_dir, agents_dir=agents_dir)
    config.autonet.enabled = False
    config.voice.enabled = False
    return Runtime(EventBus(), data_dir=data_dir, config=config)


def _service(rt, *, process="shared", max_sessions=0, code=TOOL, name="echo"):
    """A session tool and the session service it backs."""
    tool = rt.tool_store.register(
        name=f"{name}_server", description="Echoes bytes.", input_schema=SCHEMA,
        author="user", code=code,
        session={"protocol": {"id": "echo/1", "spec": "ab" * 32},
                 "max_sessions": max_sessions, "process": process,
                 "reports_usage": ["output_tokens"]})
    res = rt.service_store.register(
        name=name, description="Bytes back.", mode="session", author="user",
        backing_tool=tool["digest"], prices={"output_tokens": 1}, voucher=100,
        measures=[{"measure": "output_tokens", "quantity": 1}],
        settings={"model": "echo-large"})
    return rt.service_store.get(res["digest"])


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(service_runner, "SUPERVISE_INTERVAL", 0.2)
    monkeypatch.setattr(service_runner, "RESTART_BASE", 1.0)


@pytest.fixture
async def runner(tmp_path, fast):
    rt = _make_runtime(tmp_path)
    r = runner_for(rt)
    assert runner_for(rt) is r
    await r.start()
    try:
        yield r
    finally:
        await r.stop()


async def _ask(lease, data: bytes, *, conn=None):
    """Send ``data`` to the tool through ``lease`` and read the answer."""
    reader, writer = conn or await asyncio.open_connection(lease.host, lease.port)
    writer.write(data)
    await writer.drain()
    out = await asyncio.wait_for(reader.read(65536), 10)
    if conn is None:
        writer.close()
    return out


async def _whoami(lease):
    return json.loads(await _ask(lease, b"whoami"))


async def _until(cond, timeout=15.0):
    for _ in range(int(timeout / 0.05)):
        if cond():
            return True
        await asyncio.sleep(0.05)
    return cond()


async def test_a_shared_tool_serves_every_session_from_one_process(runner):
    rec = _service(runner._runtime)
    first = await runner.lease(rec)
    assert isinstance(first, Lease), first
    second = await runner.lease(rec)
    assert isinstance(second, Lease)
    assert (first.port, first.token) == (second.port, second.token)
    me = await _whoami(first)
    assert me["settings"] == {"model": "echo-large"}       # AUTONET_SERVICE_SETTINGS
    assert (await _whoami(second))["pid"] == me["pid"]
    assert await _ask(first, b"hello") == b"hello"
    assert runner._sessions[rec.digest] == 2
    await first.release()
    await first.release()                                   # given back once
    await second.release()
    assert runner._sessions[rec.digest] == 0
    proc = runner._servers[rec.digest].proc
    assert proc.returncode is None                          # kept running
    await runner.stop()
    assert proc.returncode is not None
    # A service whose tool serves no sessions is refused.
    assert "does not serve sessions" in await runner.lease(
        types.SimpleNamespace(backing_tool="0" * 64, digest="1" * 64))


async def test_a_crashed_shared_tool_is_restarted_and_the_error_recorded(runner):
    rec = _service(runner._runtime)
    lease = await runner.lease(rec)
    assert isinstance(lease, Lease), lease
    old_pid = (await _whoami(lease))["pid"]
    proc = runner._servers[rec.digest].proc
    try:
        await _ask(lease, b"crash")
    except Exception:  # noqa: BLE001 (the connection dies with it)
        pass
    await lease.release()
    assert await _until(lambda: proc.returncode is not None)
    # Within the backoff an opening is refused, not a restart of its own.
    refused = await runner.lease(rec)
    assert isinstance(refused, str) and "restarting" in refused
    assert any("exited with code 3" in e["error"] for e in runner.errors(rec.digest))
    # The supervisor restarts it once the backoff has passed.
    assert await _until(lambda: runner._running(runner._servers[rec.digest]), 20)
    lease = await runner.lease(rec)
    assert isinstance(lease, Lease), lease
    assert (await _whoami(lease))["pid"] != old_pid
    await lease.release()
    assert runner._servers[rec.digest].failures == 0
    log = runner._dir / "service_errors.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any(r["digest"] == rec.digest and "code 3" in r["error"] for r in rows)


async def test_a_tool_that_will_not_start_is_refused_and_recorded(runner):
    rec = _service(runner._runtime, code="import sys\nsys.exit(5)\n", name="broken")
    out = await runner.lease(rec)
    assert isinstance(out, str) and "not available" in out and "code 5" in out
    assert runner._sessions.get(rec.digest, 0) == 0
    assert any("code 5" in e["error"] for e in runner.errors(rec.digest))
    assert runner.errors("f" * 64) == []


async def test_a_per_session_tool_gets_a_process_per_session(runner):
    rec = _service(runner._runtime, process="per_session")
    a, b = await runner.lease(rec), await runner.lease(rec)
    assert isinstance(a, Lease) and isinstance(b, Lease)
    assert a.port != b.port and a.token != b.token
    assert (await _whoami(a))["pid"] != (await _whoami(b))["pid"]
    proc_a = runner._leased[a.token].proc
    await a.release()
    assert proc_a.returncode is not None                     # stopped with its session
    assert a.token not in runner._tokens and a.token not in runner._leased
    assert await _ask(b, b"still here") == b"still here"
    proc_b = runner._leased[b.token].proc
    await runner.stop()                                      # and with the runner
    assert proc_b.returncode is not None
    assert rec.digest not in runner._servers                 # never a shared one


@pytest.mark.parametrize("process", ["shared", "per_session"])
async def test_max_sessions_refuses_past_the_cap(runner, process):
    rec = _service(runner._runtime, process=process, max_sessions=1)
    # Two openings at once: one place, so one lease.
    leases = await asyncio.gather(runner.lease(rec), runner.lease(rec))
    granted = [x for x in leases if isinstance(x, Lease)]
    refused = [x for x in leases if isinstance(x, str)]
    assert len(granted) == 1 and len(refused) == 1 and "full" in refused[0]
    assert "full" in await runner.lease(rec)
    await granted[0].release()
    again = await runner.lease(rec)
    assert isinstance(again, Lease), again
    await again.release()
    assert runner._sessions[rec.digest] == 0


async def test_usage_reports_reach_the_session_that_watches_them(runner):
    rec = _service(runner._runtime)
    lease = await runner.lease(rec)
    assert isinstance(lease, Lease), lease
    reader, writer = await asyncio.open_connection(lease.host, lease.port)
    local_port = writer.get_extra_info("sockname")[1]
    seen = []
    runner.watch(lease.token, local_port, lambda m, n: seen.append((m, n)))
    assert await _ask(lease, b"x" * 300, conn=(reader, writer)) == b"x" * 300
    assert await _until(lambda: seen == [("output_tokens", 300)])
    # Another connection's usage goes to its own watcher, not this one.
    assert await _ask(lease, b"yy") == b"yy"
    await asyncio.sleep(0.5)
    assert seen == [("output_tokens", 300)]
    runner.unwatch(lease.token, local_port)
    assert await _ask(lease, b"z", conn=(reader, writer)) == b"z"
    await asyncio.sleep(0.5)
    assert seen == [("output_tokens", 300)]
    writer.close()
    await lease.release()


async def _http(port, method, path, body=b"", length=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    n = len(body) if length is None else length
    writer.write(f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {n}\r\n\r\n"
                 .encode() + body)
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), 10)
    writer.close()
    return int(line.split()[1])


async def test_bad_usage_reports_are_refused(runner):
    rec = _service(runner._runtime)
    lease = await runner.lease(rec)
    assert isinstance(lease, Lease), lease
    seen = []
    runner.watch(lease.token, 5555, lambda m, n: seen.append((m, n)))
    port, path = runner._usage_port, f"/usage/{lease.token}"

    def body(obj):
        return json.dumps(obj).encode()
    assert await _http(port, "POST", path, body(
        {"connection": 5555, "input_tokens": 4, "output_tokens": 6})) == 204
    assert await _http(port, "POST", path, body(
        [{"connection": 5555, "output_tokens": 1}, {"connection": 5556, "output_tokens": 9},
         {"connection": "x", "output_tokens": 9},
         {"connection": 5555, "output_tokens": -3, "input_tokens": "lots"}, "junk"])) == 204
    assert seen == [("input_tokens", 4), ("output_tokens", 6), ("output_tokens", 1)]
    assert await _http(port, "GET", path) == 400
    assert await _http(port, "POST", "/elsewhere", body({"connection": 5555})) == 400
    assert await _http(port, "POST", "/usage/" + "0" * 32, body(
        {"connection": 5555, "output_tokens": 1})) == 404
    assert await _http(port, "POST", path, b"{not json") == 400
    assert await _http(port, "POST", path, length=MAX_REPORT_BYTES + 1) == 400
    assert seen == [("input_tokens", 4), ("output_tokens", 6), ("output_tokens", 1)]
    await lease.release()


def test_runner_for_keeps_one_runner_per_runtime(tmp_path):
    rt = types.SimpleNamespace(_config=types.SimpleNamespace(data_dir=tmp_path))
    r = runner_for(rt)
    assert isinstance(r, ServiceRunner) and runner_for(rt) is r
    assert r._dir == tmp_path / "services"
