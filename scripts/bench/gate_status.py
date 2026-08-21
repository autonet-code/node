#!/usr/bin/env python3
"""One-shot: print the daemon's SDK-event rate-limit cache (no API call)."""
import asyncio
import json
import sys
import uuid

import websockets


async def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:7700"
    async with websockets.connect(url, open_timeout=10) as ws:
        mid = uuid.uuid4().hex
        await ws.send(json.dumps({"type": "provider_rate_limits", "msg_id": mid}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 15))
            if isinstance(msg, dict) and msg.get("msg_id") == mid:
                print(json.dumps(msg.get("result", {}), indent=1))
                return


asyncio.run(main())
