"""Orchestrator tools — Runtime operations exposed as LLM tool calls.

Each tool is a pair: a ToolDefinition (schema for the LLM) and an async
executor function that performs the operation against the Runtime.

The cognitive step's multi-turn loop calls these executors when the LLM
emits tool_use blocks.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from ..models import (
    AgentDefinition,
    AgentMode,
    AgentStatus,
    ExecutionStatus,
    HeartbeatConfig,
    InboxMessage,
    MessagePriority,
    MessageType,
    PlanningTask,
    StepDefinition,
    StepType,
    TaskStatus,
    TaskType,
)
from ..agent_registry import DelegateRegistry, DelegateStatus
from ..config import save_connector_to_config, remove_connector_from_config
from ..connectors_manager import ConnectorSpec
from ..delegate_prompts import build_delegate_prompt
from ..events import Event, EventType
from ..loader import delete_agent_dir, save_agent
from ..providers.base import ToolDefinition

if TYPE_CHECKING:
    from ..runtime import Runtime

log = logging.getLogger(__name__)

# Type for tool executor functions: (runtime, input_dict) -> result_dict
ToolExecutor = Callable[["Runtime", dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


# ---------------------------------------------------------------------------
# Tool definitions (schemas for the LLM)
# ---------------------------------------------------------------------------

_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="list_agents",
        description="List all registered agents with their current status, schedule, and concurrency settings.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="get_agent",
        description="Get detailed information about a specific agent including its step pipeline and configuration.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent's unique ID."},
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="update_agent",
        description=(
            "Update a registered agent's configuration. Can modify system_prompt, schedule, "
            "heartbeat, description, name, max_turns, or model. Changes are persisted to YAML."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent to update."},
                "system_prompt": {"type": "string", "description": "New system prompt (cognitive agents)."},
                "schedule": {"type": ["string", "null"], "description": "New schedule interval (e.g. '5m', '1h') or null to remove."},
                "heartbeat": {
                    "type": ["object", "null"],
                    "description": "Heartbeat config or null to remove.",
                    "properties": {
                        "interval": {"type": "string"},
                        "on_complete": {"type": "string", "enum": ["notify_parent", "self_deactivate"]},
                    },
                },
                "description": {"type": "string"},
                "name": {"type": "string"},
                "max_turns": {"type": "integer"},
                "model": {"type": "string", "description": "Model override (e.g. 'claude-opus-4-6')."},
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="create_agent",
        description=(
            "Create and register a new agent. Supports two modes: "
            "'pipeline' (default) with deterministic steps, or 'cognitive' for autonomous LLM sessions. "
            "Pipeline agents are created in REGISTERED state — call activate_agent afterwards. "
            "Cognitive agents with a prompt are automatically activated and triggered immediately."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique agent ID (short, lowercase, no spaces)."},
                "name": {"type": "string", "description": "Human-readable agent name."},
                "description": {"type": "string", "description": "What this agent does."},
                "mode": {
                    "type": "string",
                    "enum": ["pipeline", "cognitive"],
                    "description": "Agent mode. 'pipeline' (default) for step sequences, 'cognitive' for autonomous LLM sessions.",
                    "default": "pipeline",
                },
                "steps": {
                    "type": "array",
                    "description": "Ordered list of steps (required for pipeline mode, ignored for cognitive).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Step name."},
                            "type": {
                                "type": "string",
                                "enum": ["script", "cognitive", "message", "pull", "collect"],
                                "description": "Step type.",
                            },
                            "config": {
                                "type": "object",
                                "description": (
                                    "Step configuration. For 'script': {command, timeout}. "
                                    "For 'cognitive': {provider, model, system, prompt, max_tokens, tools}. "
                                    "For 'message': {target, mode}. "
                                    "For 'pull': {source}. "
                                    "For 'collect': {from_step, timeout}."
                                ),
                            },
                        },
                        "required": ["name", "type", "config"],
                    },
                },
                "prompt": {
                    "type": "string",
                    "description": "Task prompt for cognitive agents. The agent will work on this autonomously.",
                },
                "agent_type": {
                    "type": "string",
                    "enum": ["general", "explore", "implement", "research", "debug", "review"],
                    "description": "Cognitive agent type — determines system prompt focus.",
                    "default": "general",
                },
                "model": {
                    "type": "string",
                    "description": "Model override for cognitive agents (e.g. 'sonnet', 'opus').",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Custom system prompt for cognitive agents. If omitted, a default is generated from agent_type.",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Max LLM turns for cognitive agents. Default: 50.",
                    "default": 50,
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool surface for cognitive agents. e.g. ['sdk_builtin', 'atn_core'].",
                },
                "concurrency": {
                    "type": "integer",
                    "description": "Max parallel executions. Default: 1 (singleton).",
                    "default": 1,
                },
                "schedule": {
                    "type": "string",
                    "description": "Schedule interval (e.g. '30s', '5m', '1h'). Omit for on-demand only.",
                },
                "budgets": {
                    "type": "object",
                    "description": "Per-provider token budget limits. e.g. {'gemini': 50000}.",
                    "additionalProperties": {"type": "integer"},
                },
                "connector_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "MCP connector IDs this agent should use. Use list_connectors to see available connectors.",
                },
                "expose_as_tool": {
                    "type": "boolean",
                    "description": "If true, this pipeline agent is callable as a tool via use_tool/list_tools. Default: false.",
                    "default": False,
                },
                "tool_input_schema": {
                    "type": "object",
                    "description": "Custom JSON Schema for the tool's input. If omitted, derived from the first step's expected input.",
                },
            },
            "required": ["id", "name"],
        },
    ),
    ToolDefinition(
        name="remove_agent",
        description="Remove an agent. Kills any running executions first.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent's unique ID."},
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="activate_agent",
        description="Activate an agent — enables scheduling and inbox-triggered runs.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent's unique ID."},
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="deactivate_agent",
        description="Deactivate an agent — stops scheduling and inbox triggers.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent's unique ID."},
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="trigger_run",
        description="Trigger an immediate execution of an agent. Returns the execution ID or null if at concurrency limit.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent's unique ID."},
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="get_execution",
        description="Get an execution record including full step results and token usage. By default returns the latest for an agent. Pass execution_id to fetch a specific execution.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent's unique ID. Required if execution_id is not provided."},
                "execution_id": {"type": "string", "description": "Specific execution ID to fetch. If provided, agent_id is ignored."},
            },
        },
    ),
    ToolDefinition(
        name="get_output",
        description="Read an agent's latest output from the output store.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent's unique ID."},
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="kill_execution",
        description="Kill a specific running execution by its execution ID.",
        input_schema={
            "type": "object",
            "properties": {
                "execution_id": {"type": "string", "description": "The execution ID to kill."},
            },
            "required": ["execution_id"],
        },
    ),
    ToolDefinition(
        name="kill_agent",
        description="Kill all running executions for an agent.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent's unique ID."},
            },
            "required": ["agent_id"],
        },
    ),
    # restart_daemon — DISABLED: subprocess restart on Windows causes bridge
    # reconnection failures and orphaned processes.  Restart manually for now.
    ToolDefinition(
        name="post_message",
        description="Post a message to an agent's inbox. Use message_type 'trigger' to start an execution, 'work' to provide data.",
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target agent ID."},
                "message_type": {
                    "type": "string",
                    "enum": ["trigger", "work", "info", "alert"],
                    "description": "Message type.",
                    "default": "trigger",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Message priority.",
                    "default": "normal",
                },
                "data": {
                    "type": "object",
                    "description": "Optional data payload.",
                },
            },
            "required": ["target"],
        },
    ),
    ToolDefinition(
        name="get_snapshot",
        description="Get a full snapshot of the system state: all agents, their statuses, running executions, inbox counts.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="list_connectors",
        description="List available MCP connectors and their status. For a unified view of all tools (connectors + pipeline agents), use list_tools instead.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="add_connector",
        description=(
            "Register a new MCP connector. Supports npx (npm packages), uvx (Python packages), "
            "and local (Python scripts) modes. The connector is persisted to config and survives restarts."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique connector ID (short, lowercase, no spaces)."},
                "mode": {
                    "type": "string",
                    "enum": ["npx", "uvx", "local"],
                    "description": "Launch mode. npx for npm packages, uvx for Python packages, local for Python scripts.",
                },
                "package": {
                    "type": "string",
                    "description": "Package name (required for npx/uvx modes). e.g. '@modelcontextprotocol/server-filesystem'.",
                },
                "entry": {
                    "type": "string",
                    "description": "Entry point script (local mode only). Default: 'server.py'.",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra arguments to pass to the connector.",
                },
                "env": {
                    "type": "object",
                    "description": "Extra environment variables to set.",
                    "additionalProperties": {"type": "string"},
                },
                "env_required": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Environment variable names that must be set before the connector can start.",
                },
                "name": {"type": "string", "description": "Human-readable name for the connector."},
                "description": {"type": "string", "description": "What this connector does."},
            },
            "required": ["id", "mode"],
        },
    ),
    ToolDefinition(
        name="get_connector_tools",
        description=(
            "Get the full list of tools provided by a connector, including descriptions and input schemas. "
            "Starts the connector if it isn't running yet. Use this before use_connector to see what tools "
            "are available and what arguments they expect."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "connector_id": {"type": "string", "description": "The connector ID to inspect."},
            },
            "required": ["connector_id"],
        },
    ),
    ToolDefinition(
        name="use_connector",
        description=(
            "Call a tool on an MCP connector directly. This is the universal way to use any connector tool — "
            "browser control, filesystem access, voice, etc. The connector is started automatically if needed. "
            "Use get_connector_tools first to see available tool names and their input schemas."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "connector_id": {"type": "string", "description": "The connector to use."},
                "tool": {"type": "string", "description": "The tool name (as shown by get_connector_tools, without the mcp_ prefix)."},
                "arguments": {
                    "type": "object",
                    "description": "Tool arguments (see get_connector_tools for the schema).",
                    "additionalProperties": True,
                },
            },
            "required": ["connector_id", "tool"],
        },
    ),
    ToolDefinition(
        name="remove_connector",
        description="Remove a user-added MCP connector. Stops it if running and removes from config. Cannot remove bundled connectors.",
        input_schema={
            "type": "object",
            "properties": {
                "connector_id": {"type": "string", "description": "The connector ID to remove."},
            },
            "required": ["connector_id"],
        },
    ),
    # ------------------------------------------------------------------
    # Unified tools (connectors + pipeline-agent-tools)
    # ------------------------------------------------------------------
    ToolDefinition(
        name="list_tools",
        description=(
            "List all available tools — both MCP connectors and pipeline agents exposed as tools. "
            "Each tool has a name, description, category (connector/pipeline), and input schema. "
            "Use use_tool to call any tool by name."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["connector", "pipeline"],
                    "description": "Filter by category. Omit to list all tools.",
                },
                "include_operations": {
                    "type": "boolean",
                    "description": "Include per-tool operations (connector MCP tools, pipeline steps). Default: false.",
                    "default": False,
                },
            },
        },
    ),
    ToolDefinition(
        name="use_tool",
        description=(
            "Call any tool by its unified name — works for both MCP connectors and pipeline-agent tools. "
            "Connector tools are named 'mcp_<connector>_<tool>'. Pipeline tools are named 'pipeline_<agent_id>'. "
            "Use list_tools to discover available tools and their input schemas."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The tool's unified name (from list_tools)."},
                "arguments": {
                    "type": "object",
                    "description": "Tool arguments (see list_tools for the schema).",
                    "additionalProperties": True,
                },
            },
            "required": ["name"],
        },
    ),
    ToolDefinition(
        name="get_history",
        description="Get execution history summaries for an agent. Returns lightweight records (no full step output) for browsing previous runs.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent's unique ID."},
                "limit": {"type": "integer", "description": "Max number of records to return (default 20, max 50).", "default": 20},
            },
            "required": ["agent_id"],
        },
    ),
    # ------------------------------------------------------------------
    # Planning & goal tools
    # ------------------------------------------------------------------
    ToolDefinition(
        name="get_goals",
        description="Get the user's goals — each goal is a cognitive agent. Lists all agents with their task_prompt (goal statement) and status.",
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "completed", "paused", "abandoned"], "description": "Filter by status."},
            },
        },
    ),
    ToolDefinition(
        name="add_goal",
        description="Add a new goal by creating a cognitive agent. The goal title becomes the agent name and the description becomes its task_prompt.",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short goal title (becomes agent name)."},
                "description": {"type": "string", "description": "What success looks like (becomes agent task_prompt)."},
                "model": {"type": "string", "description": "Model for the goal agent. Defaults to sonnet."},
            },
            "required": ["title", "description"],
        },
    ),
    ToolDefinition(
        name="update_goal",
        description="Update an existing goal (agent) — change its task_prompt, name, or status.",
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "The agent ID representing this goal."},
                "status": {"type": "string", "enum": ["active", "completed", "paused", "abandoned"], "description": "New status."},
                "title": {"type": "string", "description": "Updated goal title (agent name)."},
                "description": {"type": "string", "description": "Updated goal description (agent task_prompt)."},
            },
            "required": ["goal_id"],
        },
    ),
    ToolDefinition(
        name="get_projects",
        description="Get the user's projects with status and next steps.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="add_project",
        description="Add a new project for the user.",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Project name."},
                "description": {"type": "string", "description": "What the project involves."},
                "goal_link": {"type": "string", "description": "Which goal this project supports (goal title or ID)."},
                "next_steps": {"type": "string", "description": "Immediate actions to take."},
            },
            "required": ["title", "description"],
        },
    ),
    ToolDefinition(
        name="update_project",
        description="Update an existing project's status or other fields.",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "The project's ID."},
                "status": {"type": "string", "enum": ["active", "completed", "paused", "abandoned"]},
                "next_steps": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["project_id"],
        },
    ),
    ToolDefinition(
        name="get_credit_budget",
        description="Get credit budget utilization per provider: limit, used, remaining, and auto-allocate status.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="set_credit_budget",
        description="Configure the credit budget for a provider.",
        input_schema={
            "type": "object",
            "properties": {
                "provider": {"type": "string", "description": "Provider ID (e.g. 'claude_max', 'anthropic')."},
                "token_limit": {"type": "integer", "description": "Max tokens per period."},
                "period": {"type": "string", "enum": ["daily", "weekly", "monthly"], "description": "Budget period.", "default": "monthly"},
                "auto_allocate": {"type": "boolean", "description": "Allow the planning loop to spend unused budget.", "default": True},
            },
            "required": ["provider", "token_limit"],
        },
    ),
    ToolDefinition(
        name="propose_task",
        description="Propose a planning task that advances a user's goal. The task stays in 'proposed' state until the user approves it.",
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "Which goal this task serves."},
                "title": {"type": "string", "description": "Short task title."},
                "description": {"type": "string", "description": "What the task will accomplish."},
                "task_type": {"type": "string", "enum": ["automation", "calendar", "reminder"], "description": "Type of task.", "default": "automation"},
            },
            "required": ["goal_id", "title", "description"],
        },
    ),
    ToolDefinition(
        name="list_tasks",
        description="List planning tasks, optionally filtered by status or goal.",
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["proposed", "approved", "active", "completed", "rejected"], "description": "Filter by status."},
                "goal_id": {"type": "string", "description": "Filter by goal ID."},
            },
        },
    ),
    ToolDefinition(
        name="get_user_profile",
        description="Get the user's profile summary: goals count, projects count, strengths, weaknesses, onboarding status.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # Delegation inspection tools
    ToolDefinition(
        name="delegate_status",
        description=(
            "Check the status of a running delegate sub-agent. "
            "Returns status (pending/running/completed/failed/killed) and the "
            "result if completed. Also includes output_preview (last 2000 chars "
            "of the delegate's text stream) so you can see what it's working on."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The delegate agent_id returned by delegate().",
                },
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="delegate_message",
        description=(
            "Send a message to a running delegate sub-agent. "
            "The message is injected into the delegate's conversation as a user "
            "message — it will see it on its next turn and can adjust its approach."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The delegate agent_id returned by delegate().",
                },
                "content": {
                    "type": "string",
                    "description": "The message to send to the delegate.",
                },
            },
            "required": ["agent_id", "content"],
        },
    ),
    ToolDefinition(
        name="delegate_collect",
        description=(
            "Wait for a delegate sub-agent to finish and return its result. "
            "Blocks until the delegate completes, fails, or is killed. "
            "Use this when you need the delegate's output before continuing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The delegate agent_id returned by delegate().",
                },
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="get_latest_thought",
        description=(
            "Get just the LAST conversation turn for an agent, with its timestamp. "
            "A lightweight way to check what an agent is currently doing without "
            "reading its full output."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent_id to check.",
                },
            },
            "required": ["agent_id"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool executors — async functions that perform the actual operations
# ---------------------------------------------------------------------------

async def _list_agents(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    agents = runtime.list_agents()
    return {
        "agents": [
            {
                "id": defn.id,
                "name": defn.name,
                "description": defn.description,
                "model": defn.model,
                "status": status.value,
                "schedule": defn.schedule,
                "concurrency": defn.concurrency,
                "steps": len(defn.steps),
                "budgets": defn.budgets,
                "path": str(runtime._config.agents_dir / defn.id),
            }
            for defn, status in agents
        ],
    }


async def _get_agent(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    agent_id = input["agent_id"]
    defn = runtime.get_agent(agent_id)
    if defn is None:
        return {"error": f"Agent '{agent_id}' not found."}
    status = runtime.get_status(agent_id)
    result: dict[str, Any] = {
        "id": defn.id,
        "name": defn.name,
        "description": defn.description,
        "mode": defn.mode.value,
        "model": defn.model,
        "status": status.value if status else "unknown",
        "schedule": defn.schedule,
        "concurrency": defn.concurrency,
        "budgets": defn.budgets,
        "path": str(runtime._config.agents_dir / defn.id),
        "system_prompt": defn.system_prompt or "",
        "task_prompt": defn.task_prompt or "",
    }
    # Heartbeat config
    if defn.heartbeat:
        result["heartbeat"] = {
            "interval": defn.heartbeat.interval,
            "on_complete": defn.heartbeat.on_complete,
        }
    # Next trigger countdown — heartbeat and schedule are mutually exclusive.
    # Use heartbeat_table if agent has heartbeat config, otherwise schedule_table.
    if defn.heartbeat and agent_id in runtime._heartbeat_table:
        _trigger_interval_s = runtime._heartbeat_table[agent_id]
        result["scheduling_source"] = "heartbeat"
    elif agent_id in runtime._schedule_table:
        _trigger_interval_s = runtime._schedule_table[agent_id]
        result["scheduling_source"] = "schedule"
    else:
        _trigger_interval_s = 0
    if _trigger_interval_s > 0 and status in (AgentStatus.ACTIVE, AgentStatus.RUNNING):
        last = runtime._last_idle.get(agent_id)
        if last:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            elapsed = (now - last).total_seconds()
            remaining = max(0, _trigger_interval_s - elapsed)
            result["next_trigger_in_s"] = round(remaining, 1)
            result["schedule_interval_s"] = round(_trigger_interval_s, 1)
    if defn.is_pipeline:
        result["steps"] = [
            {
                "name": s.name,
                "type": s.type.value,
                "config": s.config,
            }
            for s in defn.steps
        ]
    if defn.is_cognitive:
        result["agent_type"] = defn.agent_type
        result["max_turns"] = defn.max_turns
        result["tools"] = defn.tools
        if defn.parent_id:
            result["parent_id"] = defn.parent_id
        if defn.created_by:
            result["created_by"] = defn.created_by
        # Include output preview for running/completed cognitive agents
        output_text = runtime.get_delegate_output(agent_id)
        if output_text:
            result["output_preview"] = output_text[-2000:] if len(output_text) > 2000 else output_text
            result["output_length"] = len(output_text)
        # Children
        children = runtime.get_children(agent_id)
        if children:
            result["children"] = [
                {"id": c.id, "name": c.name, "status": (runtime.get_status(c.id) or AgentStatus.REGISTERED).value}
                for c in children
            ]
    if defn.connector_ids:
        result["connector_ids"] = defn.connector_ids
    if defn.expose_as_tool:
        result["expose_as_tool"] = True
        result["tool_name"] = f"pipeline_{defn.id}"
        if defn.tool_input_schema:
            result["tool_input_schema"] = defn.tool_input_schema
    return result


async def _update_agent(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Update a registered agent's configuration fields."""
    agent_id = input["agent_id"]
    defn = runtime.get_agent(agent_id)
    if defn is None:
        return {"error": f"Agent '{agent_id}' not found."}

    changed: list[str] = []

    if "name" in input:
        defn.name = input["name"]
        changed.append("name")
    if "description" in input:
        defn.description = input["description"]
        changed.append("description")
    if "system_prompt" in input:
        defn.system_prompt = input["system_prompt"]
        # Also sync to the cognitive step config so the LLM sees the update
        for step in defn.steps:
            if step.type == StepType.COGNITIVE and "system" in step.config:
                step.config["system"] = input["system_prompt"]
        changed.append("system_prompt")
    if "max_turns" in input:
        defn.max_turns = input["max_turns"]
        changed.append("max_turns")
    if "model" in input:
        defn.cognitive_model = input["model"]
        defn.provider = input["model"]
        changed.append("model")

    # Schedule update — mutually exclusive with heartbeat
    if "schedule" in input:
        old_schedule = defn.schedule
        defn.schedule = input["schedule"]
        if defn.schedule:
            runtime._schedule_table[agent_id] = runtime._parse_interval(defn.schedule)
            # Clear heartbeat — they are mutually exclusive
            runtime._heartbeat_table.pop(agent_id, None)
        elif agent_id in runtime._schedule_table:
            del runtime._schedule_table[agent_id]
        changed.append("schedule")

    # Heartbeat update
    if "heartbeat" in input:
        caller_id = input.get("_caller_id")
        # Access control: agents can edit their own heartbeat or their direct children's
        # Block cross-editing of unrelated agents
        if caller_id and caller_id != agent_id:
            target_defn = runtime.get_agent(agent_id)
            if target_defn and target_defn.parent_id != caller_id:
                return {"error": f"Agent '{caller_id}' can only update heartbeat for itself or its direct children."}

        hb = input["heartbeat"]
        # Treat None, empty dict {}, or {"interval": null} as "clear heartbeat"
        if hb is None or hb == {} or (isinstance(hb, dict) and hb.get("interval") is None):
            defn.heartbeat = None
            runtime._heartbeat_table.pop(agent_id, None)
        else:
            defn.heartbeat = HeartbeatConfig(
                interval=hb.get("interval", "5m"),
                on_complete=hb.get("on_complete", "notify_parent"),
            )
            runtime._heartbeat_table[agent_id] = runtime._parse_interval(defn.heartbeat.interval)
            # Clear legacy schedule — they are mutually exclusive
            runtime._schedule_table.pop(agent_id, None)
            # Always reset the idle timer so the countdown starts fresh
            from datetime import datetime, timezone
            runtime._last_idle[agent_id] = datetime.now(timezone.utc)
            # Auto-activate if agent is idle (REGISTERED/COMPLETED) so
            # the scheduler and countdown will work immediately.
            cur_status = runtime._status.get(agent_id)
            if cur_status in (AgentStatus.REGISTERED, AgentStatus.COMPLETED):
                runtime._status[agent_id] = AgentStatus.ACTIVE
        changed.append("heartbeat")

    if not changed:
        return {"agent_id": agent_id, "status": "no_changes"}

    # Persist to YAML
    try:
        save_agent(defn, runtime._config.agents_dir)
    except Exception as exc:
        log.warning("Agent updated in memory but YAML save failed: %s", exc)

    return {"agent_id": agent_id, "status": "updated", "changed": changed}


async def _create_agent(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    log.info("create_agent input: %s", json.dumps(input, default=str)[:2000])

    mode_str = input.get("mode", "pipeline")
    caller_id = input.pop("_caller_id", None)

    if mode_str == "cognitive":
        # --- Cognitive mode: autonomous LLM session ---
        prompt = input.get("prompt", "")
        agent_type = input.get("agent_type", "general")
        model = input.get("model", "")
        # parent_id is ALWAYS derived from the caller — not a user choice
        parent_id = caller_id or None

        # Auto-generate hierarchical ID if none provided
        agent_id = input.get("id", "")
        if not agent_id and parent_id:
            agent_id = runtime.generate_child_id(parent_id)
        elif not agent_id:
            agent_id = input.get("name", "agent").lower().replace(" ", "-")

        # Build system prompt — use provided or generate from agent_type
        system_prompt = input.get("system_prompt", "")
        if not system_prompt and prompt and parent_id:
            system_prompt = build_delegate_prompt(agent_type, agent_id, parent_id)

        defn = AgentDefinition(
            id=agent_id,
            name=input.get("name", "") or input.get("id", agent_id),
            mode=AgentMode.COGNITIVE,
            provider=model or runtime._config.orchestrator.model or "sonnet",
            cognitive_model=model or runtime._config.orchestrator.model or "sonnet",
            system_prompt=system_prompt,
            task_prompt=prompt,
            agent_type=agent_type,
            max_turns=input.get("max_turns", 50),
            tools=input.get("tools", ["sdk_builtin", "atn_core"]),
            concurrency=input.get("concurrency", 1),
            schedule=input.get("schedule"),
            description=input.get("description", prompt[:200] if prompt else ""),
            budgets=input.get("budgets", {}),
            connector_ids=input.get("connector_ids", []),
            parent_id=parent_id,
            created_by=caller_id or "",
        )
        try:
            aid = await runtime.register_agent(defn)
            try:
                save_agent(defn, runtime._config.agents_dir)
            except Exception as exc:
                log.warning("Agent registered but YAML save failed: %s", exc)

            # Register in DelegateRegistry for UI observability
            registry: DelegateRegistry = runtime.delegate_registry
            registry.register(
                agent_id=aid,
                parent_id=parent_id or "",
                agent_type=agent_type,
                prompt=prompt,
                title=defn.name,
            )

            # Set up done event for collect/blocking
            runtime._delegate_done[aid] = asyncio.Event()

            # If a prompt is provided, auto-activate and trigger immediately
            if prompt:
                await runtime.activate_agent(aid)
                # Post the prompt as a work message
                runtime.inbox.post(InboxMessage(
                    id=InboxMessage.generate_id(),
                    source=caller_id or "user",
                    target=aid,
                    type=MessageType.WORK,
                    priority=MessagePriority.HIGH,
                    data={"instruction": prompt},
                ))
                eid = await runtime.trigger_run(aid, source=f"agent:{caller_id}" if caller_id else "user")

                # Update delegate registry status
                registry.update_status(aid, DelegateStatus.RUNNING)

                # Emit spawn event
                node = registry.get_node(aid)
                if node:
                    await runtime.events.emit(Event(
                        type=EventType.DELEGATE_SPAWNED,
                        source=aid,
                        data=node.to_dict(),
                    ))

                return {"agent_id": aid, "status": "running", "execution_id": eid}

            return {"agent_id": aid, "status": "registered"}
        except Exception as exc:
            return {"error": str(exc)}

    else:
        # --- Pipeline mode: deterministic step sequence ---
        steps: list[StepDefinition] = []
        for s in input.get("steps", []):
            try:
                step_type = StepType(s["type"])
            except ValueError:
                return {"error": f"Invalid step type: {s['type']}. Must be one of: script, cognitive, message, pull, collect."}
            steps.append(StepDefinition(
                type=step_type,
                config=s.get("config", {}),
                name=s.get("name"),
            ))

        if not steps:
            return {"error": "Agent must have at least one step."}

        # parent_id is ALWAYS derived from the caller — not a user choice
        parent_id = caller_id or None

        defn = AgentDefinition(
            id=input["id"],
            name=input["name"],
            steps=steps,
            concurrency=input.get("concurrency", 1),
            schedule=input.get("schedule"),
            description=input.get("description", ""),
            budgets=input.get("budgets", {}),
            connector_ids=input.get("connector_ids", []),
            parent_id=parent_id,
            created_by=caller_id or "",
            expose_as_tool=bool(input.get("expose_as_tool", False)),
            tool_input_schema=input.get("tool_input_schema"),
        )

        try:
            aid = await runtime.register_agent(defn)
            # Persist to YAML so the agent survives server restarts
            try:
                save_agent(defn, runtime._config.agents_dir)
            except Exception as exc:
                log.warning("Agent registered but YAML save failed: %s", exc)
            return {"agent_id": aid, "status": "registered"}
        except Exception as exc:
            return {"error": str(exc)}


async def _remove_agent(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    agent_id = input["agent_id"]
    from . import ORCHESTRATOR_ID
    if agent_id == ORCHESTRATOR_ID:
        return {"error": "The orchestrator cannot be removed."}
    if runtime.get_agent(agent_id) is None:
        return {"error": f"Agent '{agent_id}' not found."}
    try:
        await runtime.unregister_agent(agent_id)
        # Remove YAML file so the agent doesn't reload on restart
        try:
            delete_agent_dir(agent_id, runtime._config.agents_dir)
        except Exception as exc:
            log.warning("Agent removed but YAML delete failed: %s", exc)
        return {"agent_id": agent_id, "status": "removed"}
    except Exception as exc:
        return {"error": str(exc)}


async def _activate_agent(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    agent_id = input["agent_id"]
    if runtime.get_agent(agent_id) is None:
        return {"error": f"Agent '{agent_id}' not found."}
    try:
        await runtime.activate_agent(agent_id)
        return {"agent_id": agent_id, "status": "active"}
    except Exception as exc:
        return {"error": str(exc)}


async def _deactivate_agent(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    agent_id = input["agent_id"]
    if runtime.get_agent(agent_id) is None:
        return {"error": f"Agent '{agent_id}' not found."}
    try:
        await runtime.deactivate_agent(agent_id)
        return {"agent_id": agent_id, "status": "stopped"}
    except Exception as exc:
        return {"error": str(exc)}


async def _trigger_run(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    agent_id = input["agent_id"]
    if runtime.get_agent(agent_id) is None:
        return {"error": f"Agent '{agent_id}' not found."}
    try:
        eid = await runtime.trigger_run(agent_id, source="orchestrator")
        if eid is None:
            return {"agent_id": agent_id, "error": "At concurrency limit. Execution not started."}
        return {"agent_id": agent_id, "execution_id": eid, "status": "started"}
    except Exception as exc:
        return {"error": str(exc)}


async def _get_execution(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    execution_id = input.get("execution_id")
    agent_id = input.get("agent_id")

    rec = None
    if execution_id:
        rec = runtime.execution_log.get_by_id(execution_id)
        if rec is None:
            return {"error": f"Execution '{execution_id}' not found."}
    elif agent_id:
        rec = runtime.execution_log.get_latest(agent_id)
        if rec is None:
            return {"error": f"No execution history for agent '{agent_id}'."}
    else:
        return {"error": "Provide either 'agent_id' or 'execution_id'."}

    return {
        "execution_id": rec.execution_id,
        "agent_id": rec.agent_id,
        "status": rec.status.value,
        "trigger_source": rec.trigger_source,
        "started_at": rec.started_at.isoformat() if rec.started_at else None,
        "completed_at": rec.completed_at.isoformat() if rec.completed_at else None,
        "output": rec.output,
        "error": rec.error,
        "steps": [
            {
                "name": sr.step_name,
                "type": sr.step_type.value,
                "status": sr.status.value,
                "output": sr.output,
                "error": sr.error,
                "started_at": sr.started_at.isoformat() if sr.started_at else None,
                "completed_at": sr.completed_at.isoformat() if sr.completed_at else None,
            }
            for sr in rec.step_results
        ],
        "token_usage": {
            provider: {
                "input_tokens": tu.input_tokens,
                "output_tokens": tu.output_tokens,
                "cache_read_tokens": tu.cache_read_tokens,
                "cache_creation_tokens": tu.cache_creation_tokens,
                "total": tu.total,
            }
            for provider, tu in rec.token_usage.items()
        },
    }


async def _get_output(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    agent_id = input["agent_id"]
    output = runtime.output_store.read(agent_id)
    if output is None:
        return {"error": f"No output for agent '{agent_id}'."}
    return {
        "agent_id": output.agent_id,
        "data": output.data,
        "status": output.status.value if output.status else None,
        "execution_id": output.execution_id,
        "timestamp": output.timestamp.isoformat() if output.timestamp else None,
    }


async def _kill_execution(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    eid = input["execution_id"]
    killed = await runtime.kill_execution(eid)
    if killed:
        return {"execution_id": eid, "status": "killed"}
    return {"error": f"Execution '{eid}' not found or already finished."}


async def _kill_agent(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    agent_id = input["agent_id"]
    if runtime.get_agent(agent_id) is None:
        return {"error": f"Agent '{agent_id}' not found."}
    count = await runtime.kill_agent(agent_id)
    return {"agent_id": agent_id, "killed_count": count}


## restart_daemon — DISABLED
## Subprocess restart on Windows causes bridge reconnection failures and
## orphaned processes.  The user restarts manually from the console.


async def _post_message(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    target = input["target"]
    defn = runtime.get_agent(target)
    if defn is None:
        return {"error": f"Agent '{target}' not found."}

    msg_type = MessageType(input.get("message_type") or input.get("type", "trigger"))
    priority = MessagePriority(input.get("priority", "normal"))
    instruction = (input.get("data") or {}).get("instruction", "")

    # For cognitive agents with an instruction, persist the message in
    # the agent's conversation store so it appears in its chat thread.
    if defn.mode == AgentMode.COGNITIVE and instruction:
        store = runtime.get_agent_conversation_store(target)
        store.add_user_turn(instruction)

    msg = InboxMessage(
        id=InboxMessage.generate_id(),
        source=input.get("source", "orchestrator"),
        target=target,
        type=msg_type,
        priority=priority,
        data=input.get("data", {}),
    )
    runtime.inbox.post(msg)

    # Auto-trigger idle cognitive agents so they resume without requiring
    # a separate trigger_run call.  ACTIVE means "ready but not running"
    # (e.g. after a daemon restart), COMPLETED/ERROR means "finished a
    # previous execution".  All three indicate the agent needs a new
    # execution to process the incoming message.
    from . import ORCHESTRATOR_ID
    status = runtime.get_status(target)
    execution_id = None
    if defn.mode == AgentMode.COGNITIVE and status in (
        AgentStatus.ACTIVE, AgentStatus.COMPLETED, AgentStatus.ERROR
    ):
        # Clean up stale provider from previous execution.
        # Skip the orchestrator — its provider is intentionally kept alive
        # across turns so that _session_id enables SDK session resume
        # (avoids re-sending the full conversation history every turn).
        if target != ORCHESTRATOR_ID:
            old_provider = runtime._active_providers.pop(target, None)
            if old_provider is not None:
                try:
                    await old_provider.close()
                except Exception:
                    pass
        runtime._status[target] = AgentStatus.ACTIVE
        execution_id = await runtime.trigger_run(target, source="orchestrator")

    result: dict[str, Any] = {
        "message_id": msg.id, "target": target, "type": msg_type.value,
    }
    if execution_id:
        result["execution_id"] = execution_id
        result["status"] = "triggered"
    return result


async def _get_snapshot(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    return runtime.snapshot()


async def _list_connectors(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """List connectors.  Delegates to the unified tool registry for connector category.

    Kept for backward compatibility — prefer list_tools for a unified view.
    """
    available = runtime.connectors.list_available()
    running = set(runtime.connectors.list_running())
    bundled = _get_bundled_ids()
    connectors = []
    for cid in available:
        spec = runtime.connectors.get_spec(cid)
        info: dict[str, Any] = {
            "id": cid,
            "running": cid in running,
            "bundled": cid in bundled,
        }
        if spec:
            if spec.name:
                info["name"] = spec.name
            if spec.description:
                info["description"] = spec.description
            info["mode"] = spec.mode
        if cid in running:
            session = runtime.connectors._sessions.get(cid)
            if session:
                info["tools"] = [
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get("inputSchema", {}),
                    }
                    for t in session.tools
                ]
                info["tool_count"] = len(session.tools)
        connectors.append(info)

    # Include pipeline-agent-tools count for discoverability
    from ..tool_registry import ToolCategory
    pipeline_tools = runtime.tool_registry.list_all(category=ToolCategory.PIPELINE)

    result: dict[str, Any] = {"connectors": connectors}
    if pipeline_tools:
        result["hint"] = (
            f"There are also {len(pipeline_tools)} pipeline agent(s) exposed as tools. "
            "Use list_tools to see all tools (connectors + pipelines) in a unified view."
        )
    return result


def _get_bundled_ids() -> set[str]:
    """Return the set of bundled connector IDs (for protecting against removal)."""
    try:
        from ..connectors import get_bundled_specs
        return set(get_bundled_specs().keys())
    except Exception:
        return set()


async def _add_connector(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    connector_id = input["id"]
    mode = input["mode"]

    # Validate mode-specific requirements
    if mode in ("npx", "uvx") and not input.get("package"):
        return {"error": f"Mode '{mode}' requires a 'package' field."}

    spec = ConnectorSpec(
        mode=mode,
        package=input.get("package", ""),
        entry=input.get("entry", "server.py"),
        args=input.get("args", []),
        env=input.get("env", {}),
        env_required=input.get("env_required", []),
        name=input.get("name", ""),
        description=input.get("description", ""),
    )

    runtime.connectors.register(connector_id, spec)

    # Persist to config.yaml
    spec_dict: dict[str, Any] = {"mode": mode}
    if spec.package:
        spec_dict["package"] = spec.package
    if mode == "local" and spec.entry != "server.py":
        spec_dict["entry"] = spec.entry
    if spec.args:
        spec_dict["args"] = spec.args
    if spec.env:
        spec_dict["env"] = spec.env
    if spec.env_required:
        spec_dict["env_required"] = spec.env_required

    try:
        config_path = runtime._config.data_dir / "config.yaml"
        save_connector_to_config(connector_id, spec_dict, config_path)
    except Exception as exc:
        log.warning("Connector registered but config save failed: %s", exc)

    return {"connector_id": connector_id, "status": "registered", "mode": mode}


async def _remove_connector(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    connector_id = input["connector_id"]

    # Check if it exists
    if runtime.connectors.get_spec(connector_id) is None:
        return {"error": f"Connector '{connector_id}' not found."}

    # Refuse to remove bundled connectors
    bundled = _get_bundled_ids()
    if connector_id in bundled:
        return {"error": f"Connector '{connector_id}' is bundled and cannot be removed. It auto-discovers on startup."}

    await runtime.connectors.unregister(connector_id)

    # Remove from config.yaml
    try:
        config_path = runtime._config.data_dir / "config.yaml"
        remove_connector_from_config(connector_id, config_path)
    except Exception as exc:
        log.warning("Connector unregistered but config removal failed: %s", exc)

    return {"connector_id": connector_id, "status": "removed"}


async def _get_connector_tools(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    connector_id = input["connector_id"]
    spec = runtime.connectors.get_spec(connector_id)
    if spec is None:
        return {"error": f"Connector '{connector_id}' not found."}

    # Start the connector if needed so we can discover its tools
    try:
        await runtime.connectors.ensure_started([connector_id])
    except Exception as exc:
        return {"error": f"Failed to start connector '{connector_id}': {exc}"}

    session = runtime.connectors._sessions.get(connector_id)
    if session is None:
        return {"error": f"Connector '{connector_id}' has no active session."}

    tools = []
    for t in session.tools:
        tools.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema", {}),
        })

    return {
        "connector_id": connector_id,
        "tool_count": len(tools),
        "tools": tools,
    }


async def _use_connector(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    connector_id = input["connector_id"]
    tool_name = input["tool"]
    arguments = input.get("arguments", {})

    spec = runtime.connectors.get_spec(connector_id)
    if spec is None:
        return {"error": f"Connector '{connector_id}' not found."}

    # Start the connector if needed
    try:
        await runtime.connectors.ensure_started([connector_id])
    except Exception as exc:
        return {"error": f"Failed to start connector '{connector_id}': {exc}"}

    # Call the tool
    result = await runtime.connectors.call_tool(connector_id, tool_name, arguments)
    return result


# ---------------------------------------------------------------------------
# Unified tools (connectors + pipeline-agent-tools)
# ---------------------------------------------------------------------------

async def _list_tools(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    from ..tool_registry import ToolCategory
    category_str = input.get("category")
    category: ToolCategory | None = None
    if category_str:
        try:
            category = ToolCategory(category_str)
        except ValueError:
            return {"error": f"Invalid category: '{category_str}'. Use 'connector' or 'pipeline'."}

    include_ops = bool(input.get("include_operations", False))
    tools = runtime.tool_registry.list_all(
        category=category,
        include_operations=include_ops,
    )
    return {"tools": tools, "count": len(tools)}


async def _use_tool(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    name = input.get("name", "")
    if not name:
        return {"error": "Missing required field: 'name'"}
    arguments = input.get("arguments", {})
    caller_id = input.get("_caller_id")
    return await runtime.tool_registry.call_tool(
        name, arguments, caller_id=caller_id,
    )


# ---------------------------------------------------------------------------
# Conversation management (UI-facing, not exposed to the orchestrator LLM)
# ---------------------------------------------------------------------------

async def _reset_conversation(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    session_id = runtime.conversation.reset()
    return {"status": "reset", "archived_session_id": session_id}


async def _get_conversation(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    session_id = input.get("session_id")
    if session_id:
        turns = runtime.conversation.get_session(session_id)
        if not turns:
            return {"error": f"Session '{session_id}' not found."}
        return {
            "turns": [t.to_dict() for t in turns],
            "count": len(turns),
            "total": len(turns),
        }

    limit = input.get("limit")
    offset = input.get("offset")

    if limit is not None:
        limit = min(int(limit), 200)
        offset = int(offset) if offset is not None else 0
        turns, total = runtime.conversation.get_turns_page(limit=limit, offset=offset)
        return {
            "turns": [t.to_dict() for t in turns],
            "count": len(turns),
            "total": total,
            "offset": offset,
            "has_more": (offset + len(turns)) < total,
        }

    turns = runtime.conversation.get_turns()
    return {
        "turns": [t.to_dict() for t in turns],
        "count": len(turns),
        "total": len(turns),
    }


async def _get_agent_conversation(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Get conversation history for a cognitive agent."""
    agent_id = input.get("agent_id", "")
    if not agent_id:
        return {"error": "Missing 'agent_id'"}
    defn = runtime.get_agent(agent_id)
    if defn is None:
        return {"error": f"Agent '{agent_id}' not found"}
    if defn.mode != AgentMode.COGNITIVE:
        return {"error": f"Agent '{agent_id}' is not a cognitive agent"}

    store = runtime.get_agent_conversation_store(agent_id)

    limit = input.get("limit")
    offset = input.get("offset")

    if limit is not None:
        limit = min(int(limit), 200)
        offset = int(offset) if offset is not None else 0
        turns, total = store.get_turns_page(limit=limit, offset=offset)
        return {
            "agent_id": agent_id,
            "turns": [t.to_dict() for t in turns],
            "count": len(turns),
            "total": total,
            "offset": offset,
            "has_more": (offset + len(turns)) < total,
        }

    turns = store.get_turns()
    return {
        "agent_id": agent_id,
        "turns": [t.to_dict() for t in turns],
        "count": len(turns),
        "total": len(turns),
    }


async def _send_agent_message(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Send a user message to a cognitive agent."""
    agent_id = input.get("agent_id", "")
    content = input.get("content", "")
    if not agent_id:
        return {"error": "Missing 'agent_id'"}
    if not content:
        return {"error": "Missing 'content'"}
    return await runtime.send_agent_message(agent_id, content)


async def _list_conversations(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    sessions = runtime.conversation.list_sessions()
    active_count = runtime.conversation.turn_count()
    return {
        "active_turns": active_count,
        "archived": sessions,
    }


async def _get_history(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    agent_id = input["agent_id"]
    if runtime.get_agent(agent_id) is None:
        return {"error": f"Agent '{agent_id}' not found."}
    limit = min(input.get("limit", 20), 50)
    records = runtime.execution_log.get_history(agent_id, limit=limit)
    return {
        "agent_id": agent_id,
        "count": len(records),
        "executions": [
            {
                "execution_id": rec.execution_id,
                "status": rec.status.value,
                "trigger_source": rec.trigger_source,
                "started_at": rec.started_at.isoformat() if rec.started_at else None,
                "completed_at": rec.completed_at.isoformat() if rec.completed_at else None,
                "step_count": len(rec.step_results),
                "error": rec.error,
            }
            for rec in records
        ],
    }


# ---------------------------------------------------------------------------
# Planning & goal tool executors
# ---------------------------------------------------------------------------

async def _get_goals(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Goals are agents — list all non-orchestrator agents as goals."""
    from ..orchestrator import ORCHESTRATOR_ID
    status_filter = input.get("status")
    _STATUS_MAP = {
        "active": (AgentStatus.ACTIVE, AgentStatus.RUNNING),
        "completed": (AgentStatus.COMPLETED,),
        "paused": (AgentStatus.STOPPED,),
        "abandoned": (AgentStatus.ERROR,),
    }
    goals: list[dict[str, Any]] = []
    for defn, agent_status in runtime.list_agents():
        if defn.id == ORCHESTRATOR_ID:
            continue
        # Map agent status to goal status
        if agent_status in (AgentStatus.ACTIVE, AgentStatus.RUNNING):
            goal_status = "active"
        elif agent_status == AgentStatus.COMPLETED:
            goal_status = "completed"
        elif agent_status == AgentStatus.STOPPED:
            goal_status = "paused"
        elif agent_status == AgentStatus.ERROR:
            goal_status = "abandoned"
        else:
            goal_status = agent_status.value
        if status_filter and goal_status != status_filter:
            continue
        goals.append({
            "id": defn.id,
            "title": defn.name,
            "description": defn.task_prompt or defn.description,
            "status": goal_status,
            "agent_status": agent_status.value,
            "model": defn.model,
        })
    return {"goals": goals}


async def _add_goal(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Add a goal by creating a cognitive agent."""
    title = input["title"]
    description = input["description"]
    model = input.get("model", "")

    # Generate a unique agent ID from the title
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    agent_id = f"goal-{slug}"

    # Ensure uniqueness
    if runtime.get_agent(agent_id) is not None:
        agent_id = f"{agent_id}-{__import__('uuid').uuid4().hex[:6]}"

    result = await _create_agent(runtime, {
        "id": agent_id,
        "name": title,
        "mode": "cognitive",
        "prompt": description,
        "description": f"Goal: {description}",
        "model": model or runtime._config.orchestrator.model or "claude-sonnet-4-6",
    })
    if "error" in result:
        return result
    return {"goal_id": result["agent_id"], "title": title, "description": description, "status": "added"}


async def _update_goal(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Update a goal by updating the underlying agent."""
    goal_id = input["goal_id"]
    defn = runtime.get_agent(goal_id)
    if defn is None:
        return {"error": f"Goal '{goal_id}' not found."}

    update_input: dict[str, Any] = {"agent_id": goal_id}

    if "title" in input and input["title"] is not None:
        update_input["name"] = input["title"]
    if "description" in input and input["description"] is not None:
        defn.task_prompt = input["description"]
        defn.description = f"Goal: {input['description']}"

    # Map goal status to agent operations
    status = input.get("status")
    if status == "active":
        await runtime.activate_agent(goal_id)
    elif status == "paused":
        await runtime.deactivate_agent(goal_id)
    elif status == "completed":
        await runtime.deactivate_agent(goal_id)
    elif status == "abandoned":
        await runtime.deactivate_agent(goal_id)

    result = await _update_agent(runtime, update_input)
    if "error" in result:
        return result

    return {"goal_id": goal_id, "status": "updated", "changed": result.get("changed", [])}


async def _get_projects(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    profile = runtime.user_profile.get_profile()
    return {"projects": profile.projects}


async def _add_project(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    project = runtime.user_profile.add_project({
        "title": input["title"],
        "description": input.get("description", ""),
        "goal_link": input.get("goal_link", ""),
        "next_steps": input.get("next_steps", ""),
    })
    return {"project": project, "status": "added"}


async def _update_project(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    project_id = input["project_id"]
    updates = {k: v for k, v in input.items() if k != "project_id" and v is not None}
    result = runtime.user_profile.update_project(project_id, updates)
    if result is None:
        return {"error": f"Project '{project_id}' not found."}
    return {"project": result, "status": "updated"}


async def _get_credit_budget(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    budgets = runtime.credit_budget.get_all()
    utilization = runtime.credit_budget.get_utilization()
    remaining = runtime.credit_budget.remaining_budget()
    return {
        "budgets": [
            {
                "provider": b.provider,
                "period": b.period,
                "token_limit": b.token_limit,
                "tokens_used": b.tokens_used,
                "remaining": remaining.get(b.provider, -1),
                "utilization": round(utilization.get(b.provider, 0.0), 4),
                "auto_allocate": b.auto_allocate,
            }
            for b in budgets
        ],
    }


async def _set_credit_budget(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    b = runtime.credit_budget.set_budget(
        provider=input["provider"],
        token_limit=input["token_limit"],
        period=input.get("period", "monthly"),
        auto_allocate=input.get("auto_allocate", True),
    )
    return {
        "provider": b.provider,
        "token_limit": b.token_limit,
        "period": b.period,
        "auto_allocate": b.auto_allocate,
        "status": "configured",
    }


async def _propose_task(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    task_type_str = input.get("task_type", "automation")
    try:
        task_type = TaskType(task_type_str)
    except ValueError:
        return {"error": f"Invalid task_type: {task_type_str}"}

    task = PlanningTask(
        id=PlanningTask.generate_id(),
        goal_id=input["goal_id"],
        title=input["title"],
        description=input.get("description", ""),
        task_type=task_type,
    )
    runtime.planning_tasks.append(task)
    runtime._save_planning_tasks()
    return {
        "task_id": task.id,
        "goal_id": task.goal_id,
        "title": task.title,
        "task_type": task.task_type.value,
        "status": "proposed",
    }


async def _list_tasks(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    tasks = runtime.planning_tasks
    # Apply filters
    status_filter = input.get("status")
    goal_filter = input.get("goal_id")
    if status_filter:
        tasks = [t for t in tasks if t.status.value == status_filter]
    if goal_filter:
        tasks = [t for t in tasks if t.goal_id == goal_filter]
    return {
        "tasks": [
            {
                "id": t.id,
                "goal_id": t.goal_id,
                "title": t.title,
                "description": t.description,
                "task_type": t.task_type.value,
                "status": t.status.value,
                "agent_id": t.agent_id,
                "calendar_event_id": t.calendar_event_id,
                "created_at": t.created_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in tasks
        ],
        "count": len(tasks),
    }


async def _get_user_profile(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    from ..orchestrator import ORCHESTRATOR_ID
    p = runtime.user_profile.get_profile()
    # Count goals from agent registry (non-orchestrator agents)
    goal_count = sum(1 for defn, _ in runtime.list_agents() if defn.id != ORCHESTRATOR_ID)
    return {
        "onboarding_status": p.onboarding_status.value,
        "summary": p.summary,
        "goal_count": goal_count,
        "project_count": len(p.projects),
        "strengths": p.strengths,
        "weaknesses": p.weaknesses,
        "standards_count": len(p.standards),
        "jurisdiction_id": p.jurisdiction_id,
    }


async def _approve_task(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Approve a proposed planning task (UI-facing)."""
    task_id = input.get("task_id", "")
    for t in runtime.planning_tasks:
        if t.id == task_id:
            if t.status != TaskStatus.PROPOSED:
                return {"error": f"Task '{task_id}' is not in proposed state (current: {t.status.value})."}
            t.status = TaskStatus.APPROVED
            runtime._save_planning_tasks()
            return {"task_id": task_id, "status": "approved"}
    return {"error": f"Task '{task_id}' not found."}


async def _reject_task(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Reject a proposed planning task (UI-facing)."""
    task_id = input.get("task_id", "")
    for t in runtime.planning_tasks:
        if t.id == task_id:
            if t.status != TaskStatus.PROPOSED:
                return {"error": f"Task '{task_id}' is not in proposed state (current: {t.status.value})."}
            t.status = TaskStatus.REJECTED
            runtime._save_planning_tasks()
            return {"task_id": task_id, "status": "rejected"}
    return {"error": f"Task '{task_id}' not found."}


# ---------------------------------------------------------------------------
# Sub-agent tools — scoped tool surface for child cognitive agents
# ---------------------------------------------------------------------------

# Sub-agents get a scoped subset of tools — enough for fractal recursion.
# Any cognitive agent can spawn children (via create_agent), manage them,
# and read their output.
_DELEGATE_TOOL_NAMES = {
    "create_agent",       # fractal recursion (spawn child cognitive agents)
    "delegate_status",    # check sub-agent status with timestamps
    "delegate_collect",   # wait for sub-agent result
    "delegate_message",   # send message to running sub-agent
    "get_latest_thought", # lightweight check on agent activity
    "trigger_run",    # trigger agent execution
    "get_output",     # read child agent output
    "post_message",   # communicate with other agents
    "get_snapshot",   # see system state
}


def _get_delegate_tools() -> list[dict[str, Any]]:
    """Build the tool list for a delegate sub-agent."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in _TOOLS
        if t.name in _DELEGATE_TOOL_NAMES
    ]


async def _delegate_status(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Check the status of a delegate sub-agent via the unified agent registry."""
    from datetime import datetime, timezone

    agent_id = input.get("agent_id", "")
    if not agent_id:
        return {"error": "Missing 'agent_id'."}

    # Check unified agent registry
    defn = runtime.get_agent(agent_id)
    if defn and defn.mode == AgentMode.COGNITIVE:
        status = runtime.get_status(agent_id)
        info: dict[str, Any] = {
            "agent_id": agent_id,
            "status": _agent_status_to_delegate(status),
            "title": defn.name,
            "agent_type": defn.agent_type,
        }

        # Expose timestamps from DelegateRegistry
        node = runtime.delegate_registry.get_node(agent_id)
        if node:
            info["started_at"] = node.created_at.isoformat()
            if node.completed_at:
                info["completed_at"] = node.completed_at.isoformat()
            elif node.status in (DelegateStatus.PENDING, DelegateStatus.RUNNING):
                elapsed = (datetime.now(timezone.utc) - node.created_at).total_seconds()
                info["running_duration"] = f"{elapsed:.1f}s"

        # Include current output text
        output_text = runtime.get_delegate_output(agent_id)
        if output_text:
            info["output_preview"] = output_text[-2000:] if len(output_text) > 2000 else output_text
            info["output_length"] = len(output_text)
        # Include output store data if completed
        stored = runtime.output_store.read(agent_id)
        if stored and isinstance(stored.data, dict):
            if stored.data.get("usage"):
                info["tokens_used"] = sum(stored.data["usage"].values())
            if stored.data.get("result"):
                info["result"] = stored.data["result"]
        return info

    return {"error": f"Unknown delegate: {agent_id}"}


async def _delegate_message(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Send a message to a running cognitive sub-agent."""
    agent_id = input.get("agent_id", "")
    content = input.get("content", "")
    if not agent_id:
        return {"error": "Missing 'agent_id'."}
    if not content:
        return {"error": "Missing 'content'."}

    # Try active bridge session (direct injection for immediate delivery)
    provider = runtime._active_providers.get(agent_id)
    if provider is not None:
        await provider.send_user_message(content)
        return {"status": "delivered", "agent_id": agent_id}

    # Fallback: post to inbox for agents not actively running
    # (e.g. between scheduled runs or not yet started)
    defn = runtime.get_agent(agent_id)
    if defn is not None:
        runtime.inbox.post(InboxMessage(
            id=InboxMessage.generate_id(),
            source="orchestrator",
            target=agent_id,
            type=MessageType.WORK,
            priority=MessagePriority.HIGH,
            data={"instruction": content},
        ))
        return {"status": "queued", "agent_id": agent_id, "note": "Agent not actively running; message queued in inbox."}

    return {"error": f"Delegate '{agent_id}' is not running."}


async def _delegate_collect(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Wait for a cognitive sub-agent to finish, then return its result.

    Reads from the unified output store and execution records rather than
    a separate delegate results dict.
    """
    agent_id = input.get("agent_id", "")
    if not agent_id:
        return {"error": "Missing 'agent_id'."}

    # Check if agent is known
    defn = runtime.get_agent(agent_id)
    done_event = runtime._delegate_done.get(agent_id)

    if defn is None and done_event is None:
        return {"error": f"Unknown delegate: {agent_id}"}

    # Already finished? Check agent status.
    status = runtime.get_status(agent_id)
    if status in (AgentStatus.COMPLETED, AgentStatus.ERROR, AgentStatus.STOPPED):
        runtime._delegate_done.pop(agent_id, None)
        return _build_collect_result(runtime, agent_id)

    # Wait for completion
    if done_event is not None:
        await done_event.wait()
        runtime._delegate_done.pop(agent_id, None)
        return _build_collect_result(runtime, agent_id)

    # No done event — already collected or unknown state
    return _build_collect_result(runtime, agent_id)


def _build_collect_result(runtime: "Runtime", agent_id: str) -> dict[str, Any]:
    """Build a collect result from the output store and execution records."""
    # Try output store first (populated on successful completion)
    stored = runtime.output_store.read(agent_id)
    if stored and isinstance(stored.data, dict):
        result: dict[str, Any] = {
            "agent_id": agent_id,
            "status": "completed",
            "result": stored.data.get("result", ""),
        }
        if stored.data.get("usage"):
            result["usage"] = stored.data["usage"]
        return result

    # Fall back to execution log for failures/interrupts
    status = runtime.get_status(agent_id)
    rec = runtime.execution_log.get_latest(agent_id)
    if rec is not None:
        result_text = ""
        if isinstance(rec.output, dict):
            result_text = rec.output.get("result", "")
        status_str = (
            "completed" if rec.status == ExecutionStatus.COMPLETED
            else "interrupted" if rec.status == ExecutionStatus.KILLED
            else "failed"
        )
        result = {
            "agent_id": agent_id,
            "status": status_str,
            "result": result_text,
        }
        if rec.error:
            result["error"] = rec.error
        if isinstance(rec.output, dict) and rec.output.get("usage"):
            result["usage"] = rec.output["usage"]
        return result

    # Last resort: delegate registry (backward compat)
    node = runtime.delegate_registry.get_node(agent_id)
    delegate_status = _agent_status_to_delegate(status)
    return {
        "agent_id": agent_id,
        "status": delegate_status,
        "result": node.result_preview if node else "",
        "error": node.error if node else None,
    }


def _agent_status_to_delegate(status: AgentStatus | None) -> str:
    """Map AgentStatus to the delegate status string convention."""
    if status is None:
        return "unknown"
    mapping = {
        AgentStatus.REGISTERED: "pending",
        AgentStatus.ACTIVE: "running",
        AgentStatus.RUNNING: "running",
        AgentStatus.STOPPED: "interrupted",
        AgentStatus.ERROR: "failed",
        AgentStatus.COMPLETED: "completed",
    }
    return mapping.get(status, status.value)


async def _get_latest_thought(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Return the last conversation turn for an agent, with timestamp."""
    agent_id = input.get("agent_id", "")
    if not agent_id:
        return {"error": "Missing 'agent_id'."}

    defn = runtime.get_agent(agent_id)
    if defn is None:
        return {"error": f"Unknown agent: {agent_id}"}

    store = runtime.get_agent_conversation_store(agent_id)
    turns, total = store.get_turns_page(limit=1, offset=0)
    if not turns:
        return {"agent_id": agent_id, "thought": None, "total_turns": 0}

    turn = turns[-1]
    content = turn.content
    if len(content) > 500:
        content = content[:500] + "…"

    # Find the execution_id — from the turn itself or the latest execution
    execution_id = turn.execution_id
    if not execution_id:
        rec = runtime.execution_log.get_latest(agent_id)
        if rec:
            execution_id = rec.execution_id

    return {
        "agent_id": agent_id,
        "role": turn.role,
        "content": content,
        "timestamp": turn.timestamp.isoformat(),
        "execution_id": execution_id,
        "total_turns": total,
    }


# ---------------------------------------------------------------------------
# Registry: name -> (ToolDefinition, executor)
# ---------------------------------------------------------------------------

_EXECUTORS: dict[str, ToolExecutor] = {
    "list_agents": _list_agents,
    "get_agent": _get_agent,
    "update_agent": _update_agent,
    "create_agent": _create_agent,
    "remove_agent": _remove_agent,
    "activate_agent": _activate_agent,
    "deactivate_agent": _deactivate_agent,
    "trigger_run": _trigger_run,
    "get_execution": _get_execution,
    "get_output": _get_output,
    "kill_execution": _kill_execution,
    "kill_agent": _kill_agent,
    # "restart_daemon": disabled — see comment above
    "post_message": _post_message,
    "get_snapshot": _get_snapshot,
    "list_connectors": _list_connectors,
    "add_connector": _add_connector,
    "get_connector_tools": _get_connector_tools,
    "use_connector": _use_connector,
    "remove_connector": _remove_connector,
    # Unified tools
    "list_tools": _list_tools,
    "use_tool": _use_tool,
    "get_history": _get_history,
    # Planning & goal tools
    "get_goals": _get_goals,
    "add_goal": _add_goal,
    "update_goal": _update_goal,
    "get_projects": _get_projects,
    "add_project": _add_project,
    "update_project": _update_project,
    "get_credit_budget": _get_credit_budget,
    "set_credit_budget": _set_credit_budget,
    "propose_task": _propose_task,
    "list_tasks": _list_tasks,
    "get_user_profile": _get_user_profile,
    # Delegation
    "delegate_status": _delegate_status,
    "delegate_message": _delegate_message,
    "delegate_collect": _delegate_collect,
    "get_latest_thought": _get_latest_thought,
    # Conversation management (UI-facing, not in orchestrator's tool list)
    "reset_conversation": _reset_conversation,
    "get_conversation": _get_conversation,
    "list_conversations": _list_conversations,
    # Agent conversation (UI-facing — universal chat for cognitive agents)
    "get_agent_conversation": _get_agent_conversation,
    "send_agent_message": _send_agent_message,
    # Task management (UI-facing, not in orchestrator's tool list)
    "approve_task": _approve_task,
    "reject_task": _reject_task,
}


def get_tool_definitions() -> list[ToolDefinition]:
    """Return all orchestrator tool definitions (for the LLM)."""
    return list(_TOOLS)


def get_tool_definitions_for_bridge() -> list[dict[str, Any]]:
    """Return tool definitions in bridge-compatible format (plain dicts).

    The bridge expects: {name, description, input_schema} for each tool.
    These are converted to Zod schemas at runtime in the TypeScript bridge.
    """
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in _TOOLS
    ]


def get_tool_executor(name: str) -> ToolExecutor | None:
    """Return the executor function for a tool by name."""
    return _EXECUTORS.get(name)


async def execute_tool(
    name: str,
    input: dict[str, Any],
    runtime: Runtime,
    *,
    caller_id: str | None = None,
) -> dict[str, Any]:
    """Execute a tool by name.  Returns the result dict.

    ``caller_id`` identifies the agent invoking the tool (used by delegate/
    create_agent to set parent_id for fractality).  When None, the caller is
    assumed to be the orchestrator.
    """
    executor = _EXECUTORS.get(name)
    if executor is None:
        return {"error": f"Unknown tool: {name}"}
    # Inject caller context so tools that need it can derive parent_id
    if caller_id is not None:
        input = {**input, "_caller_id": caller_id}
    try:
        return await executor(runtime, input)
    except Exception as exc:
        log.exception("Tool execution error: %s", name)
        return {"error": f"Tool '{name}' failed: {exc}"}
