"""Unified Tool Registry — connectors and pipeline-agent-tools in one place.

Everything is a tool.  Connectors already speak MCP.  Pipeline agents with
``expose_as_tool=True`` are wrapped so they look like MCP tools too.

The registry provides:
  - list_all()  — merged list of connector tools + pipeline-agent tools
  - get_tool()  — look up a tool by name (for routing)
  - call_tool() — execute any tool (connector → MCP call, pipeline → run pipeline)

Designed for future deferred loading: tools can be added/removed at runtime
via register()/unregister(), and callers can be notified of changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .models import AgentDefinition, AgentMode, StepType

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger(__name__)


class ToolCategory(Enum):
    CONNECTOR = "connector"
    PIPELINE = "pipeline"


@dataclass
class UnifiedTool:
    """A tool in the unified registry — wraps both connectors and pipeline agents."""
    name: str
    description: str
    category: ToolCategory
    input_schema: dict[str, Any]
    # For connector tools
    connector_id: str | None = None
    connector_tool_name: str | None = None
    # For pipeline-agent tools
    agent_id: str | None = None


class ToolRegistry:
    """Unified registry of all available tools (connectors + pipeline agents).

    Reads live state from the Runtime's ConnectorManager and agent registry.
    Pipeline agents with ``expose_as_tool=True`` are surfaced as tools.

    This class is stateless — it queries the Runtime on each call, so it
    always reflects the current state.  Future deferred-loading support can
    add caching and change notifications without breaking the interface.
    """

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_all(
        self,
        *,
        category: ToolCategory | None = None,
        include_operations: bool = False,
    ) -> list[dict[str, Any]]:
        """Return all available tools as dicts.

        Args:
            category: Filter by category (connector/pipeline). None = all.
            include_operations: If True, include per-tool operations (connector
                tools list their MCP operations, pipeline tools list their steps).

        Returns:
            List of tool descriptors, each with: name, description, category,
            input_schema, and optionally operations.
        """
        tools: list[dict[str, Any]] = []

        if category is None or category == ToolCategory.CONNECTOR:
            tools.extend(self._list_connector_tools(include_operations))

        if category is None or category == ToolCategory.PIPELINE:
            tools.extend(self._list_pipeline_tools(include_operations))

        return tools

    def get_tool(self, name: str) -> UnifiedTool | None:
        """Look up a tool by its unified name.

        Connector tools are named: ``tool_<connector_id>_<tool_name>``
        Pipeline tools are named: ``pipeline_<agent_id>``
        """
        # Check pipeline tools first (cheaper lookup)
        if name.startswith("pipeline_"):
            agent_id = name[len("pipeline_"):]
            defn = self._runtime._agents.get(agent_id)
            if defn and defn.expose_as_tool and defn.is_pipeline:
                return UnifiedTool(
                    name=name,
                    description=defn.description or f"Pipeline agent: {defn.name}",
                    category=ToolCategory.PIPELINE,
                    input_schema=self._derive_input_schema(defn),
                    agent_id=agent_id,
                )
            return None

        # Check connector tools
        parsed = self._runtime.connectors.parse_tool_name(name)
        if parsed:
            connector_id, tool_name = parsed
            spec = self._runtime.connectors.get_spec(connector_id)
            return UnifiedTool(
                name=name,
                description=f"[{connector_id}] {tool_name}",
                category=ToolCategory.CONNECTOR,
                input_schema={"type": "object", "properties": {}},
                connector_id=connector_id,
                connector_tool_name=tool_name,
            )

        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        caller_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a tool by its unified name.

        Routes to either the ConnectorManager (MCP call) or the pipeline
        executor depending on the tool's category.

        For pipeline tools that take time, the result is returned when the
        pipeline completes.  If ``caller_id`` is provided and the pipeline
        takes a while, a completion message is posted to the caller's inbox.
        """
        tool = self.get_tool(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}

        if tool.category == ToolCategory.CONNECTOR:
            return await self._call_connector_tool(tool, arguments)
        elif tool.category == ToolCategory.PIPELINE:
            return await self._call_pipeline_tool(tool, arguments, caller_id=caller_id)
        else:
            return {"error": f"Unsupported tool category: {tool.category}"}

    # ------------------------------------------------------------------
    # Connector tools (private)
    # ------------------------------------------------------------------

    def _list_connector_tools(self, include_operations: bool) -> list[dict[str, Any]]:
        """List all connector-provided tools."""
        tools: list[dict[str, Any]] = []
        available = self._runtime.connectors.list_available()
        running = set(self._runtime.connectors.list_running())

        for cid in available:
            spec = self._runtime.connectors.get_spec(cid)
            connector_info: dict[str, Any] = {
                "name": f"connector_{cid}",
                "description": (spec.description if spec and spec.description
                                else f"MCP connector: {spec.name or cid}" if spec else cid),
                "category": ToolCategory.CONNECTOR.value,
                "connector_id": cid,
                "running": cid in running,
            }

            if include_operations and cid in running:
                session = self._runtime.connectors._sessions.get(cid)
                if session:
                    connector_info["operations"] = [
                        {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "input_schema": t.get("inputSchema", {}),
                        }
                        for t in session.tools
                    ]
                    connector_info["operation_count"] = len(session.tools)

            tools.append(connector_info)

        return tools

    async def _call_connector_tool(
        self, tool: UnifiedTool, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Route a tool call to the connector's MCP session."""
        assert tool.connector_id and tool.connector_tool_name
        # Ensure the connector is running
        try:
            await self._runtime.connectors.ensure_started([tool.connector_id])
        except Exception as exc:
            return {"error": f"Failed to start connector '{tool.connector_id}': {exc}"}

        return await self._runtime.connectors.call_tool(
            tool.connector_id, tool.connector_tool_name, arguments,
        )

    # ------------------------------------------------------------------
    # Pipeline-agent tools (private)
    # ------------------------------------------------------------------

    def _list_pipeline_tools(self, include_operations: bool) -> list[dict[str, Any]]:
        """List pipeline agents that are exposed as tools."""
        tools: list[dict[str, Any]] = []

        for agent_id, defn in self._runtime._agents.items():
            if not defn.expose_as_tool or not defn.is_pipeline:
                continue

            tool_info: dict[str, Any] = {
                "name": f"pipeline_{agent_id}",
                "description": defn.description or f"Pipeline agent: {defn.name}",
                "category": ToolCategory.PIPELINE.value,
                "agent_id": agent_id,
                "input_schema": self._derive_input_schema(defn),
            }

            if include_operations:
                tool_info["operations"] = [
                    {
                        "name": s.name or f"step_{i}",
                        "type": s.type.value,
                    }
                    for i, s in enumerate(defn.steps)
                ]

            tools.append(tool_info)

        return tools

    async def _call_pipeline_tool(
        self,
        tool: UnifiedTool,
        arguments: dict[str, Any],
        *,
        caller_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a pipeline agent as a tool call.

        Triggers the pipeline with the provided arguments as inbox data,
        waits for completion, and returns the pipeline's output.
        """
        assert tool.agent_id
        rt = self._runtime

        defn = rt._agents.get(tool.agent_id)
        if defn is None:
            return {"error": f"Agent '{tool.agent_id}' not found"}

        # Post arguments as a work message to the agent's inbox
        from .models import InboxMessage, MessageType, MessagePriority
        msg = InboxMessage(
            id=InboxMessage.generate_id(),
            source=caller_id or "tool_registry",
            target=tool.agent_id,
            type=MessageType.WORK,
            priority=MessagePriority.NORMAL,
            data=arguments,
        )
        rt.inbox.post(msg)

        # Trigger the pipeline run
        eid = await rt.trigger_run(tool.agent_id, source=f"tool:{caller_id or 'direct'}")
        if eid is None:
            return {"error": f"Agent '{tool.agent_id}' is at concurrency limit"}

        # Wait for the execution to complete
        record = rt._executions.get(eid)
        if record is None:
            return {"error": f"Execution '{eid}' not found"}

        # Poll until the execution finishes (the task runs in the background)
        task = rt._tasks.get(eid)
        if task is not None:
            try:
                await task
            except Exception:
                pass  # errors are captured in the record

        # Re-fetch the record (may have been updated)
        record = rt.execution_log.get(eid) or rt._executions.get(eid)
        if record is None:
            return {"error": f"Execution record '{eid}' lost"}

        from .models import ExecutionStatus
        result: dict[str, Any] = {
            "execution_id": eid,
            "status": record.status.value,
        }

        if record.status == ExecutionStatus.COMPLETED:
            result["output"] = record.output
        elif record.status == ExecutionStatus.FAILED:
            result["error"] = record.error
        else:
            result["status_detail"] = f"Execution ended with status: {record.status.value}"

        # Notify caller via inbox if specified
        if caller_id and caller_id in rt._agents:
            notify_msg = InboxMessage(
                id=InboxMessage.generate_id(),
                source=tool.agent_id,
                target=caller_id,
                type=MessageType.INFO,
                priority=MessagePriority.NORMAL,
                data={
                    "tool_call_complete": True,
                    "tool_name": tool.name,
                    "execution_id": eid,
                    "status": record.status.value,
                    "output": record.output if record.status == ExecutionStatus.COMPLETED else None,
                },
            )
            rt.inbox.post(notify_msg)

        return result

    # ------------------------------------------------------------------
    # Schema derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_input_schema(defn: AgentDefinition) -> dict[str, Any]:
        """Derive the MCP-compatible input schema for a pipeline-agent tool.

        Uses ``tool_input_schema`` if set, otherwise infers from the first
        step's expected input.
        """
        if defn.tool_input_schema:
            return defn.tool_input_schema

        # Try to derive from the first step
        if defn.steps:
            first = defn.steps[0]
            if first.type == StepType.SCRIPT:
                # Script steps accept a data payload that becomes stdin / env vars
                return {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Input data passed to the pipeline as inbox work message.",
                            "additionalProperties": True,
                        },
                    },
                }
            elif first.type == StepType.COGNITIVE:
                return {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Additional prompt/instructions for the cognitive step.",
                        },
                        "data": {
                            "type": "object",
                            "description": "Structured data to include in the prompt context.",
                            "additionalProperties": True,
                        },
                    },
                }

        # Fallback: generic object schema
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "Input data for the pipeline.",
                    "additionalProperties": True,
                },
            },
        }
