"""
CDP Relay Server

Sits between Browser Use and the Chrome extension:

  Browser Use  <--CDP WebSocket-->  Relay (:9222)  <--WebSocket-->  Extension
                                                                       |
                                                                 chrome.debugger
                                                                       |
                                                                  Your real Chrome

Also serves /json/* HTTP endpoints that Browser Use expects.
"""

import asyncio
import json
import logging
import sys
import uuid

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from aiohttp import WSMsgType, web

# File-based debug logging
_log = logging.getLogger("relay")
_log.setLevel(logging.DEBUG)
_fh = logging.FileHandler("relay_debug.log", mode="w", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S"))
_log.addHandler(_fh)

# -- Relay state ---------------------------------------------------------------


class CDPRelay:
    def __init__(self):
        self.extension_ws = None  # single extension connection
        self.browser_clients = []  # Browser Use WebSocket clients
        self.pending_requests = {}  # cmd_id -> client_ws (for routing responses)
        self.pending_http = {}  # req_id -> asyncio.Future (for HTTP endpoints)
        self.training_subscribers = []  # WebSocket clients subscribed to training frames

    @property
    def extension_connected(self):
        return self.extension_ws is not None and not self.extension_ws.closed

    # -- Extension WebSocket ----------------------------------------------------

    async def handle_extension_ws(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)

        self.extension_ws = ws
        print("[Relay] Extension connected")

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                await self._on_extension_msg(data)
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSED):
                break

        self.extension_ws = None
        print("[Relay] Extension disconnected")
        return ws

    async def _on_extension_msg(self, data):
        msg_type = data.get("type")
        _log.debug("EXT>> %s", json.dumps(data, default=str)[:500])

        if msg_type == "cdp_response":
            cmd_id = data.get("id")
            client_ws = self.pending_requests.pop(cmd_id, None)
            if client_ws and not client_ws.closed:
                response = {"id": cmd_id}
                if "error" in data:
                    response["error"] = data["error"]
                else:
                    response["result"] = data.get("result", {})
                if "sessionId" in data:
                    response["sessionId"] = data["sessionId"]
                try:
                    _log.debug("->BRW response %s", json.dumps(response, default=str)[:500])
                    await client_ws.send_json(response)
                except Exception:
                    pass
            else:
                _log.debug("WARN no client for response id=%s", cmd_id)

        elif msg_type == "cdp_event":
            event = {"method": data["method"], "params": data.get("params", {})}
            if "sessionId" in data:
                event["sessionId"] = data["sessionId"]
            _log.debug("->BRW event %s", json.dumps(event, default=str)[:500])
            for client in list(self.browser_clients):
                try:
                    await client.send_json(event)
                except Exception as e:
                    _log.debug("WARN failed to send event: %s", e)

        elif msg_type == "training_frame":
            # Forward training frames to all training subscribers
            for sub in list(self.training_subscribers):
                try:
                    await sub.send_json(data)
                except Exception:
                    pass

        elif msg_type == "training_config_response":
            req_id = data.get("id")
            future = self.pending_http.pop(req_id, None)
            if future and not future.done():
                future.set_result(data)

        elif msg_type in ("tabs_response", "close_tab_response", "new_tab_response"):
            req_id = data.get("id")
            future = self.pending_http.pop(req_id, None)
            if future and not future.done():
                future.set_result(data)

    # -- Browser Use WebSocket (CDP protocol) -----------------------------------

    async def handle_browser_ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.browser_clients.append(ws)
        print("[Relay] Browser Use client connected")

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                await self._on_browser_msg(data, ws)
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSED):
                break

        if ws in self.browser_clients:
            self.browser_clients.remove(ws)
        print("[Relay] Browser Use client disconnected")
        return ws

    async def _on_browser_msg(self, data, client_ws):
        if not self.extension_connected:
            if "id" in data:
                await client_ws.send_json(
                    {"id": data["id"], "error": {"message": "Extension not connected", "code": -32000}}
                )
            return

        cmd_id = data.get("id")
        method = data.get("method", "?")
        sid = data.get("sessionId", "")
        _log.debug("BRW>> id=%s %s sid=%s", cmd_id, method, sid)

        if cmd_id is not None:
            self.pending_requests[cmd_id] = client_ws

        await self.extension_ws.send_json(
            {
                "type": "cdp_command",
                "id": cmd_id,
                "method": method,
                "params": data.get("params", {}),
                "sessionId": data.get("sessionId"),
            }
        )

    # -- HTTP endpoints (/json/*) -----------------------------------------------

    async def handle_json_version(self, request):
        return web.json_response(
            {
                "Browser": "Chrome (via Agent CDP Relay)",
                "Protocol-Version": "1.3",
                "User-Agent": "Agent CDP Relay/1.0",
                "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser",
                "extensionConnected": self.extension_connected,
            }
        )

    async def handle_json_list(self, request):
        return web.json_response(await self._ask_extension("get_tabs", {}))

    async def handle_json_new(self, request):
        url = request.query_string or "about:blank"
        result = await self._ask_extension("new_tab", {"url": url})
        return web.json_response(result)

    async def handle_json_close(self, request):
        tab_id = request.match_info.get("tab_id")
        await self._ask_extension("close_tab", {"tabId": tab_id})
        return web.json_response("Target is closing")

    # -- Training data WebSocket --------------------------------------------------

    async def handle_training_ws(self, request):
        """WebSocket endpoint for Autonet nodes to receive training frames."""
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)

        self.training_subscribers.append(ws)
        print("[Relay] Training subscriber connected")

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                # Training subscribers can send config commands
                data = json.loads(msg.data)
                if data.get("type") == "training_config" and self.extension_connected:
                    req_id = str(uuid.uuid4())
                    data["id"] = req_id
                    await self.extension_ws.send_json(data)
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSED):
                break

        if ws in self.training_subscribers:
            self.training_subscribers.remove(ws)
        print("[Relay] Training subscriber disconnected")
        return ws

    async def handle_training_config(self, request):
        """HTTP endpoint to configure training capture."""
        body = await request.json()
        result = await self._ask_extension("training_config", body)
        return web.json_response(result)

    async def _ask_extension(self, msg_type, extra):
        if not self.extension_connected:
            return []

        req_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self.pending_http[req_id] = future

        await self.extension_ws.send_json({"type": msg_type, "id": req_id, **extra})

        try:
            data = await asyncio.wait_for(future, timeout=5)
            return data.get("result", data)
        except asyncio.TimeoutError:
            return []


# -- App setup -----------------------------------------------------------------


def create_app():
    relay = CDPRelay()
    app = web.Application()

    # HTTP
    app.router.add_get("/json/version", relay.handle_json_version)
    app.router.add_get("/json/list", relay.handle_json_list)
    app.router.add_put("/json/new", relay.handle_json_new)
    app.router.add_get("/json/new", relay.handle_json_new)
    app.router.add_get("/json/close/{tab_id}", relay.handle_json_close)

    # WebSocket
    app.router.add_get("/extension", relay.handle_extension_ws)
    app.router.add_get("/devtools/browser", relay.handle_browser_ws)
    app.router.add_get("/devtools/browser/{browser_id}", relay.handle_browser_ws)
    app.router.add_get("/training", relay.handle_training_ws)

    # Training config
    app.router.add_post("/training/config", relay.handle_training_config)

    return app


if __name__ == "__main__":
    app = create_app()
    print("[Relay] CDP relay starting on http://127.0.0.1:9222")
    print("[Relay] Waiting for extension to connect on ws://127.0.0.1:9222/extension ...")
    web.run_app(app, host="127.0.0.1", port=9222, print=lambda msg: print(f"[Relay] {msg}"))
