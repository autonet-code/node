"""MCP server exposing ATN agent tools — thin daemon client.

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

Guest mode: with ``ATN_MCP_TOKEN`` set (an integration token from
``atn integration-token create --agent <id>``), the proxy connects to the
daemon's integration listener instead (docs/integration_listener.md,
default ``ws://127.0.0.1:7710``) and acts as that one agent: only the
listener's allowlist is offered (probe, use, review, author, publish
tools), and what it earns is that agent's. ``ATN_MCP_URL`` overrides the
daemon URL in either mode (e.g. several daemons on one machine).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from .agent_tools import get_tool_definitions
from .ws_server import DEFAULT_PORT

log = logging.getLogger(__name__)

GUEST_TOKEN = os.environ.get("ATN_MCP_TOKEN", "").strip()
DAEMON_URL = os.environ.get("ATN_MCP_URL", "").strip() or (
    "ws://127.0.0.1:7710" if GUEST_TOKEN else f"ws://localhost:{DEFAULT_PORT}")
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
        async with _connect(websockets) as ws:
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


def _connect(websockets: Any):
    """A connection to the daemon; in guest mode it carries the bearer token."""
    headers = {"Authorization": f"Bearer {GUEST_TOKEN}"} if GUEST_TOKEN else None
    return websockets.connect(DAEMON_URL, max_size=None, open_timeout=5,
                              additional_headers=headers)


async def _guest_allowed() -> list[str]:
    """The message types the integration listener allows this token (its
    first frame, ``integration_ready``)."""
    import websockets
    try:
        async with _connect(websockets) as ws:
            async with asyncio.timeout(10):
                async for frame in ws:
                    msg = json.loads(frame)
                    if isinstance(msg, dict) and msg.get("type") == "integration_ready":
                        return list(msg.get("allowed") or [])
    except Exception as exc:
        log.warning("guest handshake with %s failed: %s", DAEMON_URL, exc)
    return []


# Allowlisted guest calls with no agent-tool definition of their own.
_GUEST_EXTRA = {
    "status": "Who this connection acts as and what it may call.",
    "network_status": "Node, chain, epoch and peer summary of the network.",
    "tool_reviews": "Reviews and the drifted position of a tool, by digest.",
}


def _tool_list(allowed: list[str] | None) -> list[types.Tool]:
    defs = get_tool_definitions()
    if allowed is None:
        return [types.Tool(name=t.name, description=t.description,
                           inputSchema=t.input_schema) for t in defs]
    keep = set(allowed)
    tools = [types.Tool(name=t.name, description=t.description,
                        inputSchema=t.input_schema)
             for t in defs if t.name in keep]
    known = {t.name for t in tools}
    for name, desc in _GUEST_EXTRA.items():
        if name in keep and name not in known:
            tools.append(types.Tool(name=name, description=desc,
                                    inputSchema={"type": "object",
                                                 "additionalProperties": True}))
    return tools


def create_server() -> Server:
    """Create the MCP server (pure proxy — no runtime state)."""
    server = Server("atn")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return _tool_list(await _guest_allowed() if GUEST_TOKEN else None)

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
        "ATN MCP proxy ready (%s), forwarding to %s (daemon not started here)",
        "guest agent" if GUEST_TOKEN else "owner", DAEMON_URL,
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
