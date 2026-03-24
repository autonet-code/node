"""MCP server exposing ATN orchestrator tools.

Starts a full ATN Runtime, loads agents from config, and serves the
orchestrator tools over the MCP stdio transport.  Any MCP client
(Claude Code, Claude Desktop, etc.) can connect and manage agents.

Usage:
    python -m atn.mcp_server

Claude Code config (~/.claude/claude_code_config.json):
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
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from .config import load_config
from .events import EventBus
from .loader import load_agents_dir
from .orchestrator.tools import execute_tool, get_tool_definitions
from .runtime import Runtime
from .ws_server import WebSocketBridge, DEFAULT_PORT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

def create_server() -> tuple[Server, dict[str, Any]]:
    """Create the MCP server and return (server, state_dict).

    The state dict holds the Runtime reference, populated after async startup.
    """
    server = Server("atn")
    state: dict[str, Any] = {"runtime": None}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tools = get_tool_definitions()
        return [
            types.Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.input_schema,
            )
            for t in tools
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        rt = state["runtime"]
        if rt is None:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Runtime not started"}),
            )]

        result = await execute_tool(name, arguments, rt)
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str),
        )]

    return server, state


async def run_server() -> None:
    """Start the Runtime and run the MCP server."""
    # Configure logging to stderr (stdout is for MCP protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    log.info("Starting ATN MCP server...")

    # Load config and start Runtime
    config = load_config()
    bus = EventBus()
    rt = Runtime(bus, data_dir=config.data_dir, config=config)
    await rt.start()

    # Load existing agent definitions
    if config.agents_dir.exists():
        agents, errors = load_agents_dir(config.agents_dir)
        for defn in agents:
            await rt.register_agent(defn)
            if defn.schedule:
                await rt.activate_agent(defn.id)
        # Note: execution history is hydrated automatically in register_agent()
        if agents:
            log.info("Loaded %d agent(s) from %s", len(agents), config.agents_dir)
        if errors:
            log.warning("Agent load errors: %d", len(errors))
    else:
        log.info("No agents directory at %s", config.agents_dir)

    # Register the orchestrator meta-agent (always present, on-demand only)
    try:
        await rt.setup_orchestrator()
        log.info("Orchestrator registered")
    except Exception as exc:
        log.warning("Failed to register orchestrator: %s", exc)

    # Start WebSocket server on the same Runtime so Flutter sees MCP changes
    ws_bridge = WebSocketBridge(rt, host="localhost", port=DEFAULT_PORT)
    try:
        await ws_bridge.start()
        log.info("WebSocket bridge started on ws://localhost:%d", DEFAULT_PORT)
    except OSError as exc:
        log.warning("WebSocket bridge failed to start (port %d in use?): %s", DEFAULT_PORT, exc)
        ws_bridge = None

    # Create MCP server and inject Runtime
    server, state = create_server()
    state["runtime"] = rt

    log.info("ATN MCP server ready — %d tools available", len(get_tool_definitions()))

    # Run MCP server over stdio
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        if ws_bridge:
            await ws_bridge.stop()
        log.info("Shutting down Runtime...")
        await rt.stop()


def main() -> None:
    """Entry point for python -m atn.mcp_server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
