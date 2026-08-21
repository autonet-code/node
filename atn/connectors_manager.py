"""MCP connector manager — launches and manages MCP server subprocesses.

Connectors are external MCP servers that provide tool capabilities to agents.
Each connector runs as a subprocess (launched via stdio), with tools discovered
dynamically via the MCP protocol.

Launch modes:
  - local:  python <entry> (bundled Python servers)
  - npx:    npx -y <package> [args...] (npm packages)
  - uvx:    uvx <package> [args...] (Python packages via uv)

Usage:
  manager = ConnectorManager(specs)
  await manager.start("filesystem")          # launches subprocess, discovers tools
  tools = manager.get_tools("filesystem")    # returns ToolDefinition list
  result = await manager.call_tool("filesystem", "read_file", {"path": "/tmp/x"})
  await manager.stop("filesystem")           # kills subprocess
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from .providers.base import ToolDefinition

log = logging.getLogger(__name__)


@dataclass
class ConnectorSpec:
    """Descriptor for a connector's launch configuration and metadata."""
    mode: str = "local"           # "local" | "npx" | "uvx"
    package: str = ""             # npm/pypi package name (npx/uvx modes)
    entry: str = "server.py"      # entry point (local mode only)
    command: str | None = None    # explicit command override
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    env_required: list[str] = field(default_factory=list)
    # Metadata (optional, for display/discovery)
    name: str = ""                # human-readable name
    description: str = ""         # what this connector does


@dataclass
class ConnectorSession:
    """Live MCP session state for a running connector."""
    tools: list[dict[str, Any]] = field(default_factory=list)
    # Internal MCP state — managed by context managers
    _transport_cm: Any = None     # stdio_client context manager
    _session_cm: Any = None       # ClientSession context manager
    _read_stream: Any = None
    _write_stream: Any = None
    _session: ClientSession | None = None


class ConnectorManager:
    """Manages MCP connector subprocess lifecycles.

    Owns a registry of ConnectorSpec definitions (from config) and manages
    live ConnectorSession instances.  Connectors are started lazily when
    an agent needs them and kept alive across executions — but no longer
    FOREVER: an idle connector is reaped (see ``reap_idle``).

    IDLE COST IS THE POINT. A pinned tool is a subprocess spawned per call
    that exits when it returns, so an unused pinned tool costs nothing. An
    MCP connector is the opposite: a long-lived server process that, before
    the reaper, lived until daemon shutdown (``stop_all`` was reachable only
    from Runtime.stop and manual removal). Start a Blender connector once
    and it held RAM until you restarted the daemon. ``set_tool_enabled``
    does not help — it refuses FUTURE calls, it does not reclaim a running
    server.
    """

    # Default idle window before a connector is reaped. Deliberately
    # generous: restarting costs a subprocess spawn + MCP handshake +
    # tool discovery, so thrashing a connector that is used every few
    # minutes would be worse than holding it. Configurable per daemon.
    DEFAULT_IDLE_TIMEOUT_S = 900.0

    def __init__(self, specs: dict[str, ConnectorSpec] | None = None,
                 idle_timeout_s: float | None = None) -> None:
        self._specs: dict[str, ConnectorSpec] = specs or {}
        self._sessions: dict[str, ConnectorSession] = {}
        # Per-connector runtime env vars (e.g. OAuth tokens loaded from credential store).
        # Merged into the subprocess environment at start time.
        self._runtime_env: dict[str, dict[str, str]] = {}
        # connector_id -> monotonic timestamp of last start/call. Monotonic,
        # not wall-clock: a system clock adjustment must not make a live
        # connector look idle for hours (or immortal).
        self._last_used: dict[str, float] = {}
        self._idle_timeout_s: float = (
            self.DEFAULT_IDLE_TIMEOUT_S if idle_timeout_s is None
            else float(idle_timeout_s))

    @property
    def idle_timeout_s(self) -> float:
        return self._idle_timeout_s

    def _touch(self, connector_id: str) -> None:
        """Mark a connector as used now. Called on start and on every call."""
        self._last_used[connector_id] = time.monotonic()

    def idle_seconds(self, connector_id: str) -> float | None:
        """Seconds since last use, or None if not running."""
        if connector_id not in self._sessions:
            return None
        last = self._last_used.get(connector_id)
        if last is None:
            return 0.0
        return max(0.0, time.monotonic() - last)

    async def reap_idle(self, timeout_s: float | None = None) -> list[str]:
        """Stop connectors idle longer than the timeout. Returns what was
        stopped.

        A timeout <= 0 disables reaping entirely (keep-alive-forever, the
        pre-reaper behavior) — an explicit opt-out rather than an accident.
        Never raises: a reaper fault must not take down the caller's loop.
        """
        window = self._idle_timeout_s if timeout_s is None else float(timeout_s)
        if window <= 0:
            return []
        stopped: list[str] = []
        for cid in list(self._sessions.keys()):
            idle = self.idle_seconds(cid)
            if idle is None or idle < window:
                continue
            try:
                await self.stop(cid)
                stopped.append(cid)
                log.info("Connector '%s' reaped after %.0fs idle", cid, idle)
            except Exception:  # noqa: BLE001 — one bad stop must not block others
                log.exception("Error reaping connector '%s'", cid)
        return stopped

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, connector_id: str) -> None:
        """Start a connector — launch subprocess, init MCP, discover tools.

        No-op if already running.
        """
        if connector_id in self._sessions:
            log.debug("Connector '%s' already running", connector_id)
            return

        spec = self._specs.get(connector_id)
        if spec is None:
            raise ValueError(f"Unknown connector: {connector_id}")

        # Check required env vars
        for var in spec.env_required:
            if not os.environ.get(var):
                raise RuntimeError(
                    f"Connector '{connector_id}' requires env var {var} but it is not set"
                )

        server_params = self._build_server_params(connector_id, spec)
        cs = ConnectorSession()

        try:
            # Launch the MCP subprocess via stdio_client
            cs._transport_cm = stdio_client(server_params)
            streams = await cs._transport_cm.__aenter__()
            cs._read_stream, cs._write_stream = streams

            # Create and initialize the MCP client session
            cs._session_cm = ClientSession(cs._read_stream, cs._write_stream)
            cs._session = await cs._session_cm.__aenter__()
            await cs._session.initialize()

            # Discover tools
            tools_result = await cs._session.list_tools()
            cs.tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                for t in tools_result.tools
            ]

            self._sessions[connector_id] = cs
            self._touch(connector_id)
            log.info(
                "Connector '%s' started (%d tools discovered)",
                connector_id, len(cs.tools),
            )

        except Exception:
            # Clean up on failure
            await self._cleanup_session(cs)
            raise

    async def stop(self, connector_id: str) -> None:
        """Stop a running connector."""
        cs = self._sessions.pop(connector_id, None)
        self._last_used.pop(connector_id, None)
        if cs is None:
            return
        await self._cleanup_session(cs)
        log.info("Connector '%s' stopped", connector_id)

    async def stop_all(self) -> None:
        """Stop all running connectors."""
        ids = list(self._sessions.keys())
        for cid in ids:
            try:
                await self.stop(cid)
            except Exception:
                log.exception("Error stopping connector '%s'", cid)

    async def ensure_started(self, connector_ids: list[str]) -> None:
        """Ensure all specified connectors are running.  Starts any that aren't."""
        for cid in connector_ids:
            if cid not in self._sessions:
                await self.start(cid)

    # ------------------------------------------------------------------
    # Registry (runtime add/remove)
    # ------------------------------------------------------------------

    def register(self, connector_id: str, spec: ConnectorSpec) -> None:
        """Register a new connector spec at runtime.

        If a spec with this ID already exists, it is replaced (and stopped
        if currently running).
        """
        if connector_id in self._sessions:
            log.warning(
                "Replacing running connector '%s' — stopping it first",
                connector_id,
            )
            # Can't await here; caller should stop first if needed
        self._specs[connector_id] = spec
        log.info("Connector '%s' registered (mode=%s)", connector_id, spec.mode)

    async def unregister(self, connector_id: str) -> None:
        """Unregister a connector.  Stops it if running, removes from registry."""
        if connector_id in self._sessions:
            await self.stop(connector_id)
        removed = self._specs.pop(connector_id, None)
        if removed:
            log.info("Connector '%s' unregistered", connector_id)
        else:
            log.warning("Connector '%s' was not registered", connector_id)

    def get_spec(self, connector_id: str) -> ConnectorSpec | None:
        """Return the spec for a connector, or None if not registered."""
        return self._specs.get(connector_id)

    def set_runtime_env(self, connector_id: str, env: dict[str, str]) -> None:
        """Set runtime environment variables for a connector.

        These are merged into the subprocess environment when the connector
        starts.  Used to inject credentials (OAuth tokens, API keys) that
        are loaded from the credential store at runtime.
        """
        self._runtime_env[connector_id] = env

    # ------------------------------------------------------------------
    # Tool access
    # ------------------------------------------------------------------

    def list_available(self) -> list[str]:
        """Return all configured connector IDs."""
        return list(self._specs.keys())

    def list_running(self) -> list[str]:
        """Return IDs of currently running connectors."""
        return list(self._sessions.keys())

    def get_tools(self, connector_id: str) -> list[ToolDefinition]:
        """Return tool definitions for a running connector.

        Tools are namespaced as `mcp_{connector_id}_{tool_name}` to avoid
        collisions with ATN agent tools and tools from other connectors.
        """
        cs = self._sessions.get(connector_id)
        if cs is None:
            return []
        return [
            ToolDefinition(
                name=f"mcp_{connector_id}_{t['name']}",
                description=f"[{connector_id}] {t['description']}",
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
            )
            for t in cs.tools
        ]

    def get_all_tools(self, connector_ids: list[str]) -> list[ToolDefinition]:
        """Return merged tool definitions for multiple connectors."""
        tools: list[ToolDefinition] = []
        for cid in connector_ids:
            tools.extend(self.get_tools(cid))
        return tools

    async def call_tool(
        self,
        connector_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Proxy a tool call to the connector's MCP session.

        Args:
            connector_id: The connector to call.
            tool_name: The original MCP tool name (without namespace prefix).
            arguments: Tool arguments.

        Returns:
            Result dict with content items, or error.
        """
        cs = self._sessions.get(connector_id)
        if cs is None or cs._session is None:
            return {"error": f"Connector '{connector_id}' not running"}

        # Stamp BEFORE the call, not after: a long-running tool call would
        # otherwise look idle for its whole duration and could be reaped
        # out from under itself by a concurrent sweep.
        self._touch(connector_id)
        try:
            result = await cs._session.call_tool(tool_name, arguments)
            # Extract content items
            content_items: list[dict[str, Any]] = []
            for item in result.content:
                if hasattr(item, "text"):
                    content_items.append({"type": "text", "text": item.text})
                else:
                    content_items.append({"type": str(type(item).__name__)})
            return {
                "content": content_items,
                "isError": getattr(result, "isError", False),
            }
        except Exception as exc:
            log.exception("Tool call failed: %s/%s", connector_id, tool_name)
            return {"error": str(exc), "isError": True}

    # ------------------------------------------------------------------
    # Tool routing helper
    # ------------------------------------------------------------------

    def parse_tool_name(self, namespaced_name: str) -> tuple[str, str] | None:
        """Parse a namespaced tool name into (connector_id, tool_name).

        Returns None if the name doesn't match the mcp_{connector}_{tool} pattern,
        or if the connector_id isn't known.

        Examples:
            "mcp_filesystem_read_file" -> ("filesystem", "read_file")
            "list_agents" -> None  (not a connector tool)
        """
        if not namespaced_name.startswith("mcp_"):
            return None
        rest = namespaced_name[4:]  # strip "mcp_"
        # Try each known connector ID as prefix
        for cid in self._specs:
            prefix = cid + "_"
            if rest.startswith(prefix):
                tool_name = rest[len(prefix):]
                if tool_name:
                    return (cid, tool_name)
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_server_params(
        self, connector_id: str, spec: ConnectorSpec,
    ) -> StdioServerParameters:
        """Build the StdioServerParameters for launching a connector."""
        # Merge: base OS env + spec.env (from config) + runtime_env (credentials).
        # Also merge when the connector declares env_required: those vars live in
        # the daemon's os.environ and must reach the subprocess (else a connector
        # that only needs a required env var gets an empty environment).
        runtime_env = self._runtime_env.get(connector_id, {})
        if spec.env or runtime_env or spec.env_required:
            env_extra = {**os.environ, **spec.env, **runtime_env}
        else:
            env_extra = None

        if spec.mode == "local":
            command = spec.command or sys.executable
            args = [spec.entry, *spec.args]
            return StdioServerParameters(
                command=command,
                args=args,
                env=env_extra,
            )

        elif spec.mode == "npx":
            if not spec.package:
                raise ValueError(f"Connector '{connector_id}': npx mode requires 'package'")
            return StdioServerParameters(
                command="npx",
                args=["-y", spec.package, *spec.args],
                env=env_extra,
            )

        elif spec.mode == "uvx":
            if not spec.package:
                raise ValueError(f"Connector '{connector_id}': uvx mode requires 'package'")
            return StdioServerParameters(
                command="uvx",
                args=[spec.package, *spec.args],
                env=env_extra,
            )

        else:
            raise ValueError(f"Connector '{connector_id}': unknown mode '{spec.mode}'")

    @staticmethod
    async def _cleanup_session(cs: ConnectorSession) -> None:
        """Clean up a connector session's resources."""
        if cs._session_cm is not None:
            try:
                await cs._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if cs._transport_cm is not None:
            try:
                await cs._transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
