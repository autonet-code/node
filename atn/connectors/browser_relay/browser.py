"""
Persistent CDP Client -- HTTP API for browser control.

Single entry point for the Agent CDP Relay stack. Running `python browser.py`
will auto-start the relay server and Chrome (with extension) if they aren't
already running, then serve the HTTP API on port 9223.

Usage:
    python browser.py                 # start everything, serve API
    requests.get("http://localhost:9223/tabs")
    requests.post("http://localhost:9223/eval", json={"js": "document.title"})
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import base64
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from aiohttp import WSMsgType, web
import aiohttp
import requests as http_requests  # sync requests for startup checks

log = logging.getLogger("browser")
log.setLevel(logging.DEBUG)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_h)

# -- Paths & config ------------------------------------------------------------

HERE = Path(__file__).parent
RELAY_SCRIPT = HERE / "relay.py"
EXTENSION_DIR = HERE / "extension"

RELAY_HOST = "127.0.0.1"
RELAY_PORT = 9222
RELAY_URL = f"http://{RELAY_HOST}:{RELAY_PORT}"
RELAY_WS = f"ws://{RELAY_HOST}:{RELAY_PORT}/devtools/browser"

API_PORT = 9223

CHROME_PATHS = [
    # Windows
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    # macOS
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    # Linux
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
]

_relay_proc = None
_chrome_proc = None


# -- Stack bootstrap -----------------------------------------------------------

def find_chrome() -> str:
    """Find the Chrome binary on this system."""
    for path in CHROME_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Chrome not found. Searched:\n  " + "\n  ".join(CHROME_PATHS)
    )


def is_relay_alive() -> bool:
    try:
        r = http_requests.get(f"{RELAY_URL}/json/version", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def is_extension_connected() -> bool:
    try:
        r = http_requests.get(f"{RELAY_URL}/json/version", timeout=2)
        return r.status_code == 200 and r.json().get("extensionConnected", False)
    except Exception:
        return False


def _kill_stale_relay():
    """Kill any process holding the relay port that isn't our subprocess."""
    global _relay_proc
    log.info("Killing stale relay on port %d", RELAY_PORT)
    if sys.platform == "win32":
        try:
            # Find PID holding the relay port
            out = subprocess.check_output(
                ["netstat", "-ano"], text=True, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for line in out.splitlines():
                if f":{RELAY_PORT}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    log.info("Killing stale relay process PID %d", pid)
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/F"],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
        except Exception as exc:
            log.debug("Could not kill stale relay: %s", exc)
    else:
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{RELAY_PORT}"], text=True,
            ).strip()
            for pid_str in out.splitlines():
                pid = int(pid_str)
                log.info("Killing stale relay process PID %d", pid)
                os.kill(pid, 9)
        except Exception as exc:
            log.debug("Could not kill stale relay: %s", exc)

    _relay_proc = None
    time.sleep(1)


def ensure_relay():
    """Start the relay server if it isn't already running."""
    global _relay_proc
    if is_relay_alive():
        # Relay is alive -- check if extension is connected.  If not, this is
        # a stale relay from a previous session.  Kill it and start fresh so
        # Chrome will re-launch with the extension and connect properly.
        if not is_extension_connected():
            log.warning(
                "Relay alive on port %d but extension NOT connected -- "
                "killing stale relay to start fresh", RELAY_PORT,
            )
            _kill_stale_relay()
        else:
            log.info("Relay already running on port %d", RELAY_PORT)
            return

    log.info("Starting relay server...")
    _relay_proc = subprocess.Popen(
        [sys.executable, str(RELAY_SCRIPT)],
        cwd=str(HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

    for _ in range(20):
        time.sleep(0.5)
        if is_relay_alive():
            log.info("Relay server ready on port %d", RELAY_PORT)
            return
        if _relay_proc.poll() is not None:
            out = _relay_proc.stdout.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Relay exited unexpectedly:\n{out}")

    raise TimeoutError("Relay did not start within 10 seconds")


def ensure_chrome():
    """Launch Chrome with the extension if no extension is connected."""
    global _chrome_proc
    if is_extension_connected():
        log.info("Extension already connected")
        return

    chrome = find_chrome()
    log.info("Launching Chrome: %s", chrome)

    # Patch preferences to suppress "Chrome didn't shut down correctly" bar
    try:
        pref_path = os.path.expandvars(
            r"%LOCALAPPDATA%\Google\Chrome\User Data\Default\Preferences"
        )
        if os.path.isfile(pref_path):
            with open(pref_path, "r", encoding="utf-8") as f:
                prefs = json.load(f)
            prefs.setdefault("profile", {})["exit_type"] = "Normal"
            prefs["profile"]["exited_cleanly"] = True
            with open(pref_path, "w", encoding="utf-8") as f:
                json.dump(prefs, f)
    except Exception:
        pass  # non-critical

    _chrome_proc = subprocess.Popen(
        [
            chrome,
            f"--load-extension={EXTENSION_DIR}",
            "--disable-session-crashed-bubble",
            "--hide-crash-restore-bubble",
            "--profile-directory=Default",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

    log.info("Waiting for extension to connect...")
    for i in range(30):
        time.sleep(1)
        if is_extension_connected():
            log.info("Extension connected")
            return

    raise TimeoutError(
        "Extension did not connect within 30 seconds.\n"
        "Check that the extension is enabled in chrome://extensions"
    )


def bootstrap():
    """Ensure the full stack is running: relay -> Chrome -> extension."""
    ensure_relay()
    ensure_chrome()


class BrowserClient:
    def __init__(self):
        self.ws = None
        self.cmd_id = 1
        self.pending = {}  # cmd_id -> Future
        self.session_id = None  # current attached session
        self.target_id = None  # current attached tab
        self._event_log = []  # last N events for debugging
        self._reader_task = None

    async def connect(self):
        """Connect to the relay WebSocket."""
        session = aiohttp.ClientSession()
        self.ws = await session.ws_connect(RELAY_WS, heartbeat=20)
        self._reader_task = asyncio.create_task(self._read_loop())
        log.info("Connected to relay")

    async def _read_loop(self):
        """Background task reading WebSocket messages."""
        async for msg in self.ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                cmd_id = data.get("id")
                if cmd_id is not None and cmd_id in self.pending:
                    self.pending[cmd_id].set_result(data)
                else:
                    # It's an event
                    self._event_log.append(data)
                    if len(self._event_log) > 100:
                        self._event_log = self._event_log[-50:]
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSED):
                break
        log.warning("WebSocket read loop ended")

    async def cdp(self, method, params=None, session_id=None):
        """Send a CDP command and wait for the response."""
        msg = {"id": self.cmd_id, "method": method, "params": params or {}}
        if session_id:
            msg["sessionId"] = session_id
        elif self.session_id and session_id is not False:
            msg["sessionId"] = self.session_id

        future = asyncio.get_event_loop().create_future()
        self.pending[self.cmd_id] = future
        self.cmd_id += 1

        await self.ws.send_json(msg)
        try:
            result = await asyncio.wait_for(future, timeout=30)
        finally:
            self.pending.pop(msg["id"], None)
        return result

    async def evaluate(self, js, session_id=None):
        """Evaluate JavaScript in the current session."""
        sid = session_id or self.session_id
        if not sid:
            return {"error": "No session attached. Use /attach first."}
        resp = await self.cdp(
            "Runtime.evaluate",
            {"expression": js, "awaitPromise": True, "returnByValue": True},
            session_id=sid,
        )
        result = resp.get("result", {}).get("result", {})
        if "error" in resp:
            return {"error": resp["error"]}
        return {"value": result.get("value"), "type": result.get("type")}


# -- Main (standalone HTTP API) ------------------------------------------------

if __name__ == "__main__":
    async def main():
        bootstrap()  # auto-start relay + Chrome if needed

        client = BrowserClient()
        await client.connect()

        # Clean orphaned markers from any previous session
        try:
            resp = await client.cdp("Target.getTargets", session_id=False)
            targets = resp.get("result", {}).get("targetInfos", [])
            marker = "\U0001f534 "
            orphaned = [t for t in targets if t.get("type") == "page" and t.get("title", "").startswith(marker)]
            for t in orphaned:
                tid = t["targetId"]
                try:
                    r = await client.cdp("Target.attachToTarget", {"targetId": tid, "flatten": True}, session_id=False)
                    sid = r.get("result", {}).get("sessionId")
                    if sid:
                        await client.cdp("Runtime.evaluate", {
                            "expression": '(function() { var M = "\\uD83D\\uDD34 "; if (window.__agentMarkerObserver) { window.__agentMarkerObserver.disconnect(); delete window.__agentMarkerObserver; } if (document.title.startsWith(M)) document.title = document.title.slice(M.length); })()',
                        }, session_id=sid)
                        await client.cdp("Target.detachFromTarget", {"sessionId": sid}, session_id=False)
                        log.info("Cleaned orphaned marker from tab %s (%s)", tid, t.get("title", "")[:60])
                except Exception:
                    pass
        except Exception:
            pass

        app = web.Application()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", API_PORT)
        await site.start()

        log.info("Browser API ready on http://127.0.0.1:%d", API_PORT)

        # Keep running
        while True:
            await asyncio.sleep(3600)

    asyncio.run(main())
