#!/usr/bin/env python3
"""WAN run telemetry sidecar (issue #23), run on each VM.

Every interval, for each local daemon: ask its owner listener for
network_status, epoch_status and autonet_status, take the warnings and
errors from its log since the last report, and POST one report to the hub's
ingest (https://obs.autonet.computer/ingest, bearer token). The VM's egress
proxy is used automatically (HTTPS_PROXY).

    python telemetry.py --obs https://obs.autonet.computer --token-file obs.token \\
        --daemon vm1-h1=ws://127.0.0.1:7800,container=vm1-h1 \\
        --daemon vm1-h2=ws://127.0.0.1:7801,log=/data/h2/daemon.log
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

KEEP = ("WARNING", "ERROR", "CRITICAL", "Traceback", "rewind", "anchored",
        "federated close", "Connected to", "disconnect")
STATUS_KEYS = ("status", "errors", "last_error", "paused_reason", "uptime_seconds",
               "memory_mb", "cpu_percent", "jurisdiction_key", "charter_match",
               "jurisdiction_errors")


async def ask(url: str, kind: str, timeout: float = 20.0) -> dict:
    import websockets
    mid = uuid.uuid4().hex
    try:
        async with websockets.connect(url, open_timeout=10, max_size=None) as ws:
            await ws.send(json.dumps({"type": kind, "msg_id": mid}))
            deadline = time.time() + timeout
            while time.time() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), deadline - time.time()))
                if isinstance(msg, dict) and msg.get("msg_id") == mid:
                    return msg.get("result") if msg.get("ok") else {"error": msg.get("error")}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)[:300]}
    return {"error": "timeout"}


class LogTail:
    def __init__(self, path: str = "", container: str = "") -> None:
        self.path, self.container, self.pos, self.since = path, container, 0, None

    def new_lines(self) -> list[str]:
        if self.container:
            args = ["docker", "logs", self.container]
            if self.since:
                args += ["--since", self.since]
            self.since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            r = subprocess.run(args, capture_output=True, text=True, errors="replace")
            lines = (r.stdout + r.stderr).splitlines()
        elif self.path and Path(self.path).exists():
            with open(self.path, encoding="utf-8", errors="replace") as fh:
                fh.seek(self.pos)
                lines = fh.read().splitlines()
                self.pos = fh.tell()
        else:
            lines = []
        return [ln[:400] for ln in lines if any(k in ln for k in KEEP)][-60:]


def post(obs: str, token: str, report: dict) -> str:
    req = urllib.request.Request(
        obs.rstrip("/") + "/ingest", data=json.dumps(report, default=str).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return str(r.status)
    except Exception as exc:  # noqa: BLE001
        return repr(exc)[:200]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--daemon", action="append", required=True,
                    help="name=ws://host:port[,log=path|,container=name]")
    ap.add_argument("--interval", type=float, default=60.0)
    a = ap.parse_args()
    token = Path(a.token_file).read_text(encoding="utf-8").strip()
    daemons = []
    for spec in a.daemon:
        name, _, rest = spec.partition("=")
        url, *opts = rest.split(",")
        kw = dict(o.split("=", 1) for o in opts)
        daemons.append((name, url, LogTail(kw.get("log", ""), kw.get("container", ""))))
    host = os.environ.get("HOSTNAME") or os.uname().nodename
    while True:
        for name, url, tail in daemons:
            status = await ask(url, "autonet_status")
            report = {
                "daemon": name, "host": host, "t": time.time(),
                "network": await ask(url, "network_status"),
                "epoch": await ask(url, "epoch_status"),
                "status": {k: status.get(k) for k in STATUS_KEYS} if isinstance(status, dict) else status,
                "log": tail.new_lines(),
            }
            print(name, post(a.obs, token, report), flush=True)
        await asyncio.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
