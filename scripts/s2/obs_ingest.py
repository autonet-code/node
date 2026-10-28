#!/usr/bin/env python3
"""WAN run observability ingest (issue #23), run on the hub.

VM sidecars (scripts/s2/telemetry.py) POST one JSON report per daemon per
minute to /ingest; the collector on the owner's machine GETs /latest (the
newest report per daemon) and /since?t=<unix> (every report after t). Both
need the bearer token. Reports are appended to <dir>/<daemon>.jsonl, so the
run's whole history survives restarts.

    python obs_ingest.py --port 4120 --dir ~/autonet-wan/obs --token-file ~/autonet-wan/obs.token

Stdlib only; listens on 127.0.0.1 behind nginx (obs.autonet.computer).
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_BODY = 256 * 1024
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def make_handler(store: Path, token: str):
    lock = threading.Lock()
    latest: dict[str, dict] = {}

    class Handler(BaseHTTPRequestHandler):
        def _authed(self) -> bool:
            return self.headers.get("Authorization", "") == f"Bearer {token}"

        def _send(self, code: int, body: object) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path == "/beacon":
                # The red-team canary's exfiltration target: no auth, logged
                # only, so a leak past containment shows up here.
                n = min(int(self.headers.get("Content-Length") or 0), 8192)
                body = self.rfile.read(n).decode("utf-8", errors="replace") if n else ""
                with lock:
                    with open(store / "_beacon.log", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"t": time.time(), "from": self.client_address[0],
                                             "body": body[:4000]}) + "\n")
                return self._send(200, {"ok": True})
            if urlparse(self.path).path != "/ingest" or not self._authed():
                return self._send(403, {"error": "forbidden"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._send(413, {"error": "size"})
            try:
                report = json.loads(self.rfile.read(n))
                name = str(report["daemon"])
            except Exception:  # noqa: BLE001
                return self._send(400, {"error": "bad report"})
            if not NAME_RE.match(name):
                return self._send(400, {"error": "bad daemon name"})
            report["received"] = time.time()
            with lock:
                latest[name] = report
                with open(store / f"{name}.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(report) + "\n")
            self._send(200, {"ok": True})

        def do_GET(self):  # noqa: N802
            if not self._authed():
                return self._send(403, {"error": "forbidden"})
            url = urlparse(self.path)
            if url.path == "/latest":
                with lock:
                    return self._send(200, latest)
            if url.path.startswith("/share/"):
                # Run files a VM needs (jurisdiction.json, vm_env.json, summary.json).
                name = url.path[len("/share/"):]
                f = store.parent / "share" / name
                if not NAME_RE.match(name) or not f.is_file():
                    return self._send(404, {"error": "not found"})
                return self._send(200, json.loads(f.read_text(encoding="utf-8")))
            if url.path == "/since":
                t = float((parse_qs(url.query).get("t") or ["0"])[0])
                out = []
                for f in sorted(store.glob("*.jsonl")):
                    for line in f.read_text(encoding="utf-8").splitlines():
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if r.get("received", 0) > t:
                            out.append(r)
                return self._send(200, out[-5000:])
            self._send(404, {"error": "not found"})

        def log_message(self, *args):  # quiet
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4120)
    ap.add_argument("--dir", default="~/autonet-wan/obs")
    ap.add_argument("--token-file", required=True)
    a = ap.parse_args()
    store = Path(a.dir).expanduser()
    store.mkdir(parents=True, exist_ok=True)
    token = Path(a.token_file).expanduser().read_text(encoding="utf-8").strip()
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(store, token)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
