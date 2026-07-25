"""MCP server exposing ATN orchestrator tools — thin daemon client.

Connects to a RUNNING ATN daemon over its local WebSocket (:7700) and
proxies tool calls.  It never boots a Runtime of its own: no agent
loading, no schedules, no port grabbing, no state writes.  If the daemon
isn't running, tools return an error telling the user to start it.

Usage:
    python -m atn.mcp_server

Claude Code config (.mcp.json):
    {
      "mcpServers": {
        "atn": {
          "command": "python",
          "args": ["-m", "atn.mcp_server"]
        }
      }
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from .orchestrator.tools import get_tool_definitions
from .ws_server import DEFAULT_PORT

log = logging.getLogger(__name__)

DAEMON_URL = f"ws://localhost:{DEFAULT_PORT}"
_CALL_TIMEOUT = 300.0  # seconds; some tools (trigger_run, delegate) run long

_NOT_RUNNING = (
    f"ATN daemon is not running ({DAEMON_URL}). "
    "Start it with `python -m atn` and retry."
)


async def _call_daemon(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Send one tool call to the daemon's local WS listener and await the reply.

    One connection per call — stateless and immune to stale sockets. Frames
    that aren't the reply (event broadcasts) are skipped by msg_id match.
    """
    import websockets

    mid = uuid.uuid4().hex
    payload = dict(arguments)
    payload["type"] = name
    payload["msg_id"] = mid

    try:
        async with websockets.connect(DAEMON_URL, max_size=None, open_timeout=5) as ws:
            await ws.send(json.dumps(payload))
            async with asyncio.timeout(_CALL_TIMEOUT):
                async for frame in ws:
                    try:
                        msg = json.loads(frame)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(msg, dict) and msg.get("msg_id") == mid:
                        if msg.get("ok"):
                            result = msg.get("result")
                            return result if isinstance(result, dict) else {"result": result}
                        return {"error": msg.get("error") or "daemon returned an error"}
        return {"error": "daemon closed the connection before replying"}
    except (ConnectionRefusedError, OSError):
        return {"error": _NOT_RUNNING}
    except asyncio.TimeoutError:
        return {"error": f"daemon did not reply within {_CALL_TIMEOUT:.0f}s"}
    except Exception as exc:  # ConnectionClosed et al.
        return {"error": f"daemon connection failed: {exc}"}


def create_server() -> Server:
    """Create the MCP server (pure proxy — no runtime state)."""
    server = Server("atn")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.input_schema,
            )
            for t in get_tool_definitions()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        result = await _call_daemon(name, arguments or {})
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str),
        )]

    return server


async def run_server() -> None:
    """Run the MCP stdio server (daemon is contacted per tool call)."""
    # Configure logging to stderr (stdout is for MCP protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    log.info(
        "ATN MCP proxy ready — %d tools, forwarding to %s (daemon not started here)",
        len(get_tool_definitions()), DAEMON_URL,
    )

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point for python -m atn.mcp_server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
