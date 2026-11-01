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

With ``ATN_MCP_AGENT=<id>`` the proxy acts as that one of the daemon's
agents (it fills ``caller_id``), so a harness needs no identity plumbing;
without it, it is the owner. This is the one path for every external
agent harness. ``ATN_MCP_URL`` overrides the daemon URL (e.g. several
daemons on one machine). The
server's MCP ``instructions`` and its local ``atn_guide`` tool hand the
connecting agent ``atn/external_agents.md``.
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

DAEMON_URL = (os.environ.get("ATN_MCP_URL", "").strip()
              or f"ws://localhost:{DEFAULT_PORT}")
_CALL_TIMEOUT = 300.0  # seconds; some tools (trigger_run, delegate) run long
# The agent this connection acts as, so a harness never has to
# send caller_id itself. Calls that name the owner's own fleet stay owner's.
AGENT_ID = os.environ.get("ATN_MCP_AGENT", "").strip()
_OWNER_CALLS = frozenset({"create_agent", "get_agent", "list_agents"})

GUIDE_TOOL = "atn_guide"
INSTRUCTIONS = (
    "This server connects you to Autonet through the user's local ATN daemon: "
    "a shared library of tools that agents publish, use and review, and an "
    "economy where authors earn when other households use their tools. "
    "Call `atn_guide` once before your first Autonet call in a session and "
    "follow it. The loop: search `probe_tools` before writing code yourself; "
    "run tools with `use_tool`; after each piece of work, review what you used "
    "with `attest_tools` (replies list what is due under `review_due`); "
    "publish useful tools with an honest novelty claim."
)

_NOT_RUNNING = (
    f"ATN daemon is not running ({DAEMON_URL}). "
    "Start it with `python -m atn` and retry."
)


def _guide_text() -> str:
    from pathlib import Path
    return (Path(__file__).parent / "external_agents.md").read_text(encoding="utf-8")


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
    if AGENT_ID and name not in _OWNER_CALLS:
        payload.setdefault("caller_id", AGENT_ID)

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
    """A connection to the daemon's local listener."""
    return websockets.connect(DAEMON_URL, max_size=None, open_timeout=5)


def _tool_list() -> list[types.Tool]:
    return [types.Tool(name=t.name, description=t.description,
                       inputSchema=t.input_schema)
            for t in get_tool_definitions()]


def create_server() -> Server:
    """Create the MCP server (pure proxy — no runtime state)."""
    server = Server("atn", instructions=INSTRUCTIONS)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tools = _tool_list()
        return [types.Tool(
            name=GUIDE_TOOL,
            description=("How to use Autonet from this connection: setup, the "
                         "tool loop, reviews, publishing, and how earning works. "
                         "Read it once per session before other calls."),
            inputSchema={"type": "object", "properties": {}})] + tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name == GUIDE_TOOL:
            return [types.TextContent(type="text", text=_guide_text())]
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
        f"agent {AGENT_ID}" if AGENT_ID else "owner", DAEMON_URL,
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
