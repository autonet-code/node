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
from ..runtime.provider_manager import get_model_tier, get_tier_label

if TYPE_CHECKING:
    from ..runtime import Runtime

log = logging.getLogger(__name__)


def _flat_budgets(runtime: "Runtime", defn: Any) -> dict[str, Any]:
    """Normalize an agent's budgets to a flat ``{key: int_limit}`` map for the
    config surface.

    The stored shape accepts a bare int OR a nested ``{limit, period}`` dict
    (agent_registry._parse_one_budget). Returning the raw nested shape to a
    client that expects a scalar per key makes it coerce the dict to 0 — the
    "40k → 0 tokens in the config tab" bug. Resolving through get_budget_info
    (which parses every shape to a numeric limit) guarantees a scalar limit per
    key regardless of how the budget was authored."""
    if not getattr(defn, "budgets", None):
        return {}
    try:
        info = runtime.registry.get_budget_info(defn.id)
    except Exception:
        info = {}
    if info:
        # Token limits are conceptually integers; get_budget_info resolves them
        # as floats. Emit ints so a scalar-expecting client renders cleanly.
        out: dict[str, Any] = {}
        for key, entry in info.items():
            lim = entry.get("limit", 0) or 0
            out[key] = int(lim) if float(lim).is_integer() else lim
        return out
    # Fallback: best-effort flatten without the registry (e.g. detached defn).
    flat: dict[str, Any] = {}
    for key, raw in defn.budgets.items():
        if isinstance(raw, dict):
            flat[key] = raw.get("limit", raw.get("pct", 0)) or 0
        else:
            try:
                flat[key] = int(raw)
            except (TypeError, ValueError):
                flat[key] = 0
    return flat

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
            "Update a registered agent's configuration. Supports all fields including "
            "advanced config (schedule, budgets, connectors, pipeline steps, concurrency). "
            "Changes are persisted to YAML."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent to update."},
                "system_prompt": {"type": "string", "description": "New system prompt (cognitive agents)."},
                "schedule": {"type": ["string", "null"], "description": "Schedule interval (e.g. '5m', '1h') or null to remove."},
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
                "model": {"type": "string", "description": "Model override (e.g. 'claude-opus-4-7', 'gpt-5.5')."},
                "notify_parent": {
                    "type": "boolean",
                    "description": "If false, skip auto-notification to parent on completion. Default: true.",
                },
                "wake_parent_on_child": {
                    "type": "boolean",
                    "description": (
                        "If true, a child reaching a terminal state wakes THIS "
                        "agent (as parent) with an immediate run instead of "
                        "batching the notification into its next natural run. "
                        "Default: false."
                    ),
                },
                "parent_id": {
                    "type": ["string", "null"],
                    "description": (
                        "Reparent this agent under a new parent (the 'regional "
                        "manager' path), or null to promote it to top-level. "
                        "Rejected on cycles or if the new subtree breaks spawn/"
                        "depth/budget limits."
                    ),
                },
                "concurrency": {"type": "integer", "description": "Max parallel executions."},
                "budgets": {
                    "type": "object",
                    "description": "Per-provider token budget limits. e.g. {'gemini': 50000}.",
                    "additionalProperties": {"type": "integer"},
                },
                "connector_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "MCP connector IDs this agent should use.",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool surface for cognitive agents. e.g. ['delegation', 'messaging'].",
                },
                "steps": {
                    "type": "array",
                    "description": "Pipeline steps (pipeline mode only).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": ["script", "cognitive", "message", "pull", "collect"]},
                            "config": {"type": "object"},
                        },
                        "required": ["name", "type", "config"],
                    },
                },
                "expose_as_tool": {"type": "boolean", "description": "Expose pipeline agent as a callable tool."},
                "tool_input_schema": {"type": "object", "description": "Custom JSON Schema for the tool's input."},
                "provider": {
                    "type": "string",
                    "description": (
                        "Explicit provider for this agent (e.g. 'ollama', "
                        "'anthropic', 'rpb'). Set after 'model', so it wins "
                        "when both are given."
                    ),
                },
                "service_provider": {
                    "type": ["object", "null"],
                    "description": (
                        "Bind this agent's inference to a marketplace service "
                        "you bought — the agent then thinks on that substrate "
                        "and pays each call from ITS OWN wallet (fund it "
                        "on-chain first). PARENT-ONLY: you may set this on your "
                        "DIRECT CHILDREN, never on yourself. Pass null to "
                        "unbind. Choosing an agent's substrate is the "
                        "employer's call, so an agent can never set or change "
                        "its own."
                    ),
                    "properties": {
                        "provider_address": {
                            "type": "string",
                            "description": "The serving agent's 0x address.",
                        },
                        "spec_digest": {
                            "type": "string",
                            "description": "sha256 digest of the service spec bought.",
                        },
                    },
                    "required": ["provider_address", "spec_digest"],
                },
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="create_agent",
        description=(
            "Create and register a new agent. Most agents are 'cognitive' — just provide "
            "id, name, and prompt. Use update_agent afterwards for advanced config "
            "(schedule, budgets, connectors, pipeline steps, concurrency)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique agent ID (short, lowercase, no spaces)."},
                "name": {"type": "string", "description": "Human-readable agent name."},
                "prompt": {
                    "type": "string",
                    "description": "Task prompt for cognitive agents. The agent will work on this autonomously.",
                },
                "description": {"type": "string", "description": "What this agent does."},
                "mode": {
                    "type": "string",
                    "enum": ["pipeline", "cognitive"],
                    "description": "Agent mode. Default: 'cognitive'.",
                    "default": "cognitive",
                },
                "agent_type": {
                    "type": "string",
                    "enum": ["general", "explore", "implement", "research", "debug", "review"],
                    "description": "Cognitive agent type — determines system prompt focus.",
                    "default": "general",
                },
                "model": {
                    "type": "string",
                    "description": "Model override (e.g. 'sonnet', 'opus').",
                },
                "provider": {
                    "type": "string",
                    "enum": [
                        "claude_max", "codex_max", "anthropic", "openai",
                        "gemini", "deepseek", "ollama", "rpb", "substrate",
                    ],
                    "description": (
                        "Explicit provider for this agent. Omit to route by the "
                        "model id (the daemon fails loud if a model can't be "
                        "placed rather than defaulting onto the subscription). "
                        "Set 'ollama' for local models, 'rpb' for sponsor-routed "
                        "dependents."
                    ),
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Max LLM turns. Default: 50.",
                    "default": 50,
                },
                "notify_parent": {
                    "type": "boolean",
                    "description": "If false, skip auto-notification to parent on completion. Default: true.",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Tool categories: 'delegation' (spawn/manage children), 'messaging', "
                        "'observation' (snapshots/history), 'lifecycle' (activate/kill), "
                        "'connectors', 'unified_tools', 'planning', 'budget', 'identity'. "
                        "Or 'atn_full' for all. Default: delegation + messaging + observation."
                    ),
                },
                "budgets": {
                    "type": "object",
                    "description": (
                        "Optional per-provider token cap for this child, e.g. "
                        "{'claude_max': 200000}. Omit to leave the child bounded by the "
                        "subtree's root budget (recommended for normal sub-agents)."
                    ),
                    "additionalProperties": {"type": "integer"},
                },
                "service_provider": {
                    "type": "object",
                    "description": (
                        "Bind this new child's inference to a marketplace "
                        "service you bought: it thinks on that substrate and "
                        "pays each call from ITS OWN wallet, which you fund "
                        "on-chain. Overrides 'provider'/'model'. Use this to "
                        "provision a child on purchased cognition and judge its "
                        "output from outside that substrate."
                    ),
                    "properties": {
                        "provider_address": {
                            "type": "string",
                            "description": "The serving agent's 0x address.",
                        },
                        "spec_digest": {
                            "type": "string",
                            "description": "sha256 digest of the service spec bought.",
                        },
                    },
                    "required": ["provider_address", "spec_digest"],
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
            "Discover available tools: core ATN tools, MCP connectors, and pipeline agents. "
            "Returns name + description for each. Use use_tool to call any by name. "
            "Pass include_operations=true to get full input schemas."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["connector", "pipeline", "core", "registered"],
                    "description": "Filter by category. 'core' = ATN framework tools, 'registered' = substrate-registered tools visible to you. Omit to list all.",
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
            "Call any tool by name — core ATN tools, MCP connectors, or pipeline agents. "
            "Use list_tools first to see available tools and their input schemas. "
            "REVIEW STEP: if you invoke registered (network) tools, the work item "
            "is not closed until you call attest_tools once at the end, reviewing "
            "the tools that mattered with per-charter-axis scores — reviews are "
            "the signal that routes every agent's tool discovery."
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
        name="register_tool",
        description=(
            "Author a new tool. Provide either `code` (a Python script reading JSON "
            "arguments on stdin and printing a JSON result — becomes a PINNED tool, "
            "behavior locked by content hash) or a `connector_id` (ATTESTED, "
            "connector-backed). Registration is ALWAYS private: local capability "
            "scoped to you and your superiors; only the user can grant it outside "
            "that lineage. You own what you author. Publishing to the substrate "
            "(where consensus judges it and you earn mint from standing and usage) "
            "is the separate publish_tool capability, granted case-by-case. Remote "
            "paid APIs are Services, not tools: use the services rail instead of "
            "an endpoint."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tool name (snake_case, verb-first)."},
                "description": {"type": "string", "description": "What the tool does — this is also how it's discovered, so be precise."},
                "input_schema": {
                    "type": "object",
                    "description": "JSON schema of the tool's arguments.",
                    "additionalProperties": True,
                },
                "code": {"type": "string", "description": "Python source (pinned tools). Reads JSON args on stdin, prints JSON result on stdout."},
                "connector_id": {"type": "string", "description": "MCP connector backing this tool (attested); the tool name must match a connector operation."},
                "provider": {"type": "string", "description": "External provider identity for attested tools (e.g. 'google')."},
                "dependencies": {"type": "array", "items": {"type": "string"}, "description": "Digests of published tools this tool calls at runtime (pinned only). Declaration = the runtime allowlist; nested calls run under the ORIGINAL caller's authority. Composite tools use the line-framed sandbox protocol (see docs/tool_substrate.md)."},
                "capabilities": {"type": "object", "description": "What the code needs from the host, deny-by-default: {net: bool, fs: bool, spawn: bool, env: [VAR,...], secrets: [SERVICE,...]}. On adopting daemons this IS the sandbox policy — undeclared use hard-fails, so declare honestly. `secrets` names the vault services the code reads at runtime: the daemon binds ONLY those to the tool's own process (clamped by the calling agent's allowance — declaring does not grant), and the tool reads each value from ATN_TOOL_SECRETS via its own broker session. Omit if the tool only reads stdin and writes stdout."},
                "version_of": {"type": "string", "description": "Digest of the manifest this revises (artifact lineage)."},
            },
            "required": ["name", "description", "input_schema"],
        },
    ),
    ToolDefinition(
        name="publish_tool",
        description=(
            "Publish a tool YOU authored to the substrate: its manifest becomes "
            "network-visible, debatable, and (pinned tools) mint-eligible. This is "
            "a separately granted capability — having register_tool does not imply "
            "having this. You can only publish your own tools."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "digest": {"type": "string", "description": "Manifest digest (or name/reg_ prefix) of a tool you registered."},
            },
            "required": ["digest"],
        },
    ),
    ToolDefinition(
        name="adopt_tool",
        description=(
            "Propose adopting a tool published on the network (find digests "
            "with the substrate probe). Adoption installs FOREIGN CODE on "
            "this host, so you only PROPOSE: the owner sees the manifest, "
            "its declared capabilities (the sandbox policy it will run "
            "under — deny-by-default), signature, and vet status, and "
            "approves or rejects per tool. If approved, the tool becomes "
            "callable by you and your lineage, runs contained, and its "
            "ORIGINAL author keeps earning from your attestations. Give a "
            "concrete reason: what work needs it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "digest": {"type": "string", "description": "64-hex manifest digest of the published tool."},
                "reason": {"type": "string", "description": "Why you need it — shown to the owner."},
            },
            "required": ["digest", "reason"],
        },
    ),
    ToolDefinition(
        name="vet_tool",
        description=(
            "Inspect an unreviewed published tool to price its risk before "
            "you or your owner trust it (validator role — separately "
            "granted). Call with only a digest to INSPECT: you receive the "
            "manifest and the pinned source code to read. Then call again "
            "with verdict 'pass' (code adheres to the manifest and contains "
            "no malice) or 'fail', a report stating what you checked, and "
            "optional per-charter-axis 'axes' scores (-1..+1) for what you "
            "read. Your inspection is a review with no usage receipt: it "
            "moves the tool's PUBLIC rating and charter position — weighted "
            "by YOUR reputation and credibility, so a zero-reputation "
            "inspector moves nothing. There is no royalty, no earnings, no "
            "greenlight quorum, and no gate: inspection does not unlock the "
            "tool or pay you — it is how the network prices an unreviewed "
            "tool so agents can decide whether to use it. You cannot vet "
            "your own tools."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "digest": {"type": "string", "description": "Manifest digest (64-hex) — local or network-published."},
                "verdict": {"type": "string", "enum": ["pass", "fail"], "description": "Omit to inspect; set to record your inspection review."},
                "report": {"type": "string", "description": "Required with a verdict: what you checked, what you found."},
                "axes": {
                    "type": "object",
                    "description": (
                        "Optional per-charter-axis scores from reading the code, "
                        "-1..+1. Keys: life_precious, self_preservation, "
                        "promotion_of_intelligence, evolution, correctness, "
                        "simplicity. Only include axes you can judge from the "
                        "source; these move the tool's public position/rating."
                    ),
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["digest"],
        },
    ),
    ToolDefinition(
        name="check_evidence",
        description=(
            "Verify an evidence-bearing CON against a pinned tool, then "
            "optionally back it (validator role — vetting bundle). A CON "
            "disputing a tool can carry a reproducible failing invocation "
            "(args + expected result/error). This RE-RUNS the pinned code "
            "with those args on your own daemon and reports whether the "
            "failure reproduces. If it does — and you pass the CON's node "
            "id — you post a support sprout under the CON, lending your "
            "standing to a dispute you personally reproduced. Evidence "
            "recruits verification; a non-reproducing invocation recruits "
            "no one, so your standing is never spent on a claim you could "
            "not confirm."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "manifest_digest": {"type": "string", "description": "64-hex digest of the disputed pinned tool (must be resolvable locally — adopt or register it first)."},
                "evidence": {
                    "type": "object",
                    "description": "The CON's evidence: {args_json (object or JSON string), expected_error OR expected_digest, actual_digest (optional)}.",
                    "additionalProperties": True,
                },
                "con_node_id": {"type": "string", "description": "CON node id to support if the replay confirms. Omit to only replay (diagnostic)."},
                "support": {"type": "boolean", "description": "Post a support sprout when confirmed (default true). Set false to replay without backing.", "default": True},
            },
            "required": ["manifest_digest", "evidence"],
        },
    ),
    ToolDefinition(
        name="run_trial",
        description=(
            "Run a venture's service verifier trial battery (validator role "
            "— vetting bundle). Given a venture prospectus digest (a "
            "published artifact declaring the service's expected behavior, a "
            "pre-committed black-box trial battery, and free-inference "
            "credentials), this fetches the prospectus, executes each "
            "declared trial case against the service's MCP surface, scores "
            "pass/fail against the prospectus's OWN criteria, blob-stores a "
            "trial report, and returns the verdict plus the report digest. "
            "A moat can't be read, so it is PROBED: you attest what you "
            "observed. Submit the returned attestTrial calldata/verdict "
            "on-chain to record your trial (the vault greenlight reads it)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prospectus_digest": {"type": "string", "description": "64-hex digest of the published venture_prospectus artifact."},
            },
            "required": ["prospectus_digest"],
        },
    ),
    ToolDefinition(
        name="attest_tools",
        description=(
            "Review which registered tools contributed to a piece of work you "
            "just finished. Call ONCE when you close a work item, judging the "
            "tools that helped — not after every call. Your attestation is the "
            "ONLY usage that counts toward a tool author's mint; a mechanical "
            "call receipt is worth nothing. Rate per charter axis in 'axes' "
            "when you have a real signal: signed scores in -1..+1 for any of "
            "correctness (did what it claimed, no bugs), simplicity (minimal, "
            "not over-engineered), life_precious, self_preservation, "
            "promotion_of_intelligence, evolution. Score only axes you "
            "actually observed — omit the rest. Your axis scores move the "
            "tool's position in charter space and rank it in library search "
            "(good reviews get a tool found and used more — that is how "
            "reviews pay authors); they never change the mint amount "
            "directly. Provide 'context': what you were working on "
            "(embedded to locate the work)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "judgments": {
                    "type": "array",
                    "description": "One entry per tool that contributed.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "description": "Tool name, digest, or reg_ prefix."},
                            "ok": {"type": "boolean", "description": "Did the tool serve the work?"},
                            "score": {"type": "number", "description": "Optional overall quality score 0..1."},
                            "axes": {
                                "type": "object",
                                "description": (
                                    "Optional per-charter-axis signed scores, -1..+1. "
                                    "Keys: life_precious, self_preservation, "
                                    "promotion_of_intelligence, evolution, "
                                    "correctness, simplicity. Only include axes "
                                    "you observed."
                                ),
                                "additionalProperties": {"type": "number"},
                            },
                            "note": {"type": "string", "description": "Optional reviewable justification (blob-stored)."},
                        },
                        "required": ["tool", "ok"],
                    },
                },
                "context": {
                    "type": "string",
                    "description": "What you were working on — the closed work item.",
                },
            },
            "required": ["judgments", "context"],
        },
    ),
    ToolDefinition(
        name="probe_tools",
        description=(
            "Search the network tool library semantically before building "
            "something yourself or asking for a grant. Search is UNFILTERED: "
            "a niche tool surfaces on topic match regardless of its score, "
            "and good ratings only lift it higher — so read the trust "
            "picture on each match to price risk yourself. Each match "
            "carries digest, name, description, author, 'rating' and the "
            "6-axis charter position ('axes'), plus 'review_mass' (how much "
            "earned reputation is behind the score) and 'inspections' (how "
            "many agents read the code). Read it like this: high 'rating' "
            "with high 'review_mass' = trusted by earned voice, safe to use; "
            "'review_mass' near zero = nobody has reviewed it yet, inspect "
            "it (vet_tool) or verify before trusting; a bad (negative) "
            "rating = avoid or verify. Use the digest with adopt_tool (if "
            "granted) or ask your owner to grant/adopt it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you need the tool to do.",
                },
                "k": {
                    "type": "integer",
                    "description": "Max matches (default 8).",
                },
            },
            "required": ["query"],
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
        name="get_my_budget_status",
        description=(
            "Returns the calling agent's per-provider budget caps, current "
            "usage, ancestor caps & remaining headroom, plus subscription-window "
            "utilization for any subscription provider in use. Call this when you "
            "need to decide whether to start expensive work or escalate to your "
            "parent for more headroom."
        ),
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="get_usage",
        description=(
            "Get YOUR provider usage so you can self-pace long-horizon work "
            "against the shared subscription. For Claude Max / bridge agents: "
            "5h + 7d subscription-window utilization and reset times, plus your "
            "own budget consumption. For API/local providers: cumulative session "
            "tokens vs your configured budget. Cached ~60s — safe to poll between "
            "phases without burning a probe call every turn."
        ),
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="metering_report",
        description=(
            "Daemon-wide operational metering report (admin/ops use). Reads the "
            "metering service's PERSISTED spend events + subscription snapshots "
            "and returns a deterministic rollup: per-provider time-bucketed cost "
            "series with cache-hit ratios and cost-per-ktok (a dropping cache-hit "
            "ratio or rising cost-per-ktok flags 'prompt caching broken'); "
            "per-agent burn ranking (which agents are spending); and inferred "
            "subscription quota + remaining + confidence per subscription "
            "provider. Pure over persisted ledgers — no live provider calls. Use "
            "this to detect provider anomalies, budget leaks, and quota "
            "trajectory. You interpret the numbers in prose; the tool only "
            "measures."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "window_hours": {
                    "type": "number",
                    "description": "How far back to include events. Default 24.",
                    "default": 24,
                },
                "bucket": {
                    "type": "string",
                    "enum": ["hour", "day"],
                    "description": "Time-bucket granularity for the series. Default 'hour'.",
                    "default": "hour",
                },
            },
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
    ToolDefinition(
        name="get_children_status",
        description=(
            "Get compact status for all direct children of the calling agent. "
            "Returns each child's id, name, status, turns, last_tool, and "
            "conversation_path (path to conversation JSONL or delegate output log)."
        ),
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="register_on_chain",
        description=(
            "Register a child agent on-chain in the jurisdiction's RPB contract. "
            "The daemon signs the transaction with the agent's stored private key. "
            "Returns the transaction hash on success."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The child agent to register."},
                "sponsor_address": {
                    "type": "string",
                    "description": "Optional sponsor agent address (0x...).",
                    "default": "",
                },
            },
            "required": ["agent_id"],
        },
    ),
    ToolDefinition(
        name="compact_agent",
        description=(
            "Compact a target agent's conversation to free context (§15). "
            "You may compact your own DIRECT CHILDREN; the owner may compact any "
            "agent. You can NEVER compact yourself. If the agent is running its "
            "generic loop, compaction is queued for the next iteration boundary; "
            "if it is idle, its persisted history is summarized in place and the "
            "cached provider evicted so the next run rebuilds from the compacted "
            "store. A running bridge (Claude SDK) agent cannot be compacted "
            "mid-run and returns 'unsupported_while_running'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "The agent to compact."},
            },
            "required": ["agent_id"],
        },
    ),
    # ------------------------------------------------------------------
    # Services rail (docs/services_market.md) — the remote-API market.
    # A Service is a paid remote API another agent (its human benefactor)
    # offers: general-purpose, priced per work item, settled in ATN via
    # Substrate.payForService. Services get NO substrate standing, mint,
    # or verdict claims — trust is behavioral (identity, atomic payment).
    # ------------------------------------------------------------------
    ToolDefinition(
        name="find_services",
        description=(
            "Discover services offered on the network — remote paid APIs other "
            "agents sell (NOT tools: services run on the provider's daemon and "
            "settle in ATN, they earn no substrate standing). Lists on-chain "
            "services most-recent first: service_id, provider address, ask "
            "(ATN price per work item), spec digest, and whether it's active. "
            "Pass 'query' to filter by a substring against locally-known service "
            "names/descriptions. To invoke one: pay_for_service to the provider, "
            "then request_service with the payment proof. Read-only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional substring/topic filter over locally-known service metadata.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max services to return (default 25).",
                },
            },
        },
    ),
    ToolDefinition(
        name="register_service",
        description=(
            "Author a service YOU offer: a remote paid API other agents can "
            "invoke, priced per work item in ATN. This builds and persists the "
            "local service spec AND registers it on-chain in the ServiceRegistry "
            "under your agent address (you become the provider). Requires you to "
            "be registered on-chain (register_on_chain first) — the contract's "
            "onlyAgent gate rejects unregistered callers. A service needs a "
            "backing tool to actually fulfil requests: pass 'backing_tool' (the "
            "digest of a registered tool you own) so incoming service_requests "
            "dispatch to it. Returns {service_id, tx_hash, spec_digest}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Service name (how it's discovered — be precise)."},
                "description": {"type": "string", "description": "What the service does."},
                "ask_amount": {
                    "type": "integer",
                    "description": "Price per work item in ATN base units (uint). This is the on-chain ask and the payment minimum the gate enforces.",
                },
                "input_schema": {
                    "type": "object",
                    "description": "JSON schema of the service's request arguments.",
                    "additionalProperties": True,
                },
                "backing_tool": {
                    "type": "string",
                    "description": "Digest of a registered tool you own that fulfils requests (a Service is a tool the owner chose to sell).",
                },
                "output_schema": {
                    "type": "object",
                    "description": "Optional JSON schema of the service's result.",
                    "additionalProperties": True,
                },
                "endpoint_hint": {"type": "string", "description": "Optional reachability hint (advisory)."},
                "image_uri": {
                    "type": "string",
                    "description": "Optional https URL of a banner image for the "
                                   "marketplace card (advisory, display-only). "
                                   "Use a real, stable image you host or "
                                   "uploaded; without it the card shows a "
                                   "default banner.",
                },
            },
            "required": ["name", "description", "ask_amount"],
        },
    ),
    ToolDefinition(
        name="pay_for_service",
        description=(
            "Pay a service provider in ATN before invoking their service. This "
            "signs a Substrate.payForService transfer from YOUR agent key to the "
            "provider (taking the network service fee) and returns {tx_hash, "
            "request_id}. Hand BOTH the tx_hash and request_id to the provider "
            "when you call request_service — the provider verifies the payment "
            "on-chain against the service ask before serving. 'amount' is the "
            "gross ATN (>= the service ask); the provider receives net of fee. "
            "Generate one payment per request: a request_id is single-use "
            "(replay-protected provider-side)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "The provider's agent address (0x...)."},
                "amount": {"type": "integer", "description": "Gross ATN base units to pay (>= the service ask)."},
                "request_id": {
                    "type": "string",
                    "description": "Optional bytes32 hex request id; a fresh one is generated if omitted.",
                },
            },
            "required": ["recipient", "amount"],
        },
    ),
    ToolDefinition(
        name="request_service",
        description=(
            "Invoke a service on the provider's daemon (cross-daemon call). "
            "Resolves the provider's WS endpoint from the on-chain agent "
            "registry, connects, and sends the request with your payment proof "
            "(the tx_hash + request_id from pay_for_service). Pass the "
            "'service_id''s spec_digest as 'service_id' is looked up on chain, "
            "and the request 'payload' (matching the service's input schema). "
            "Returns the provider's result, or a clear error if the provider "
            "published no endpoint or the payment failed verification."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "provider_address": {"type": "string", "description": "The provider agent's address (0x...)."},
                "service_id": {"type": "string", "description": "The service's spec digest (from find_services)."},
                "payload": {
                    "type": "object",
                    "description": "The request arguments (per the service input schema).",
                    "additionalProperties": True,
                },
                "tx_hash": {"type": "string", "description": "The payForService tx hash from pay_for_service."},
                "request_id": {"type": "string", "description": "The request_id from pay_for_service (bytes32 hex)."},
            },
            "required": ["provider_address", "service_id", "payload", "tx_hash", "request_id"],
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
                "budgets": _flat_budgets(runtime, defn),
                "path": str(runtime._config.agents_dir / defn.id),
                "registered_on_chain": defn.identity.registered_on_chain if defn.identity else False,
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
        "budgets": _flat_budgets(runtime, defn),
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
    result["notify_parent"] = defn.notify_parent
    # Which substrate this agent thinks on. `provider` was previously invisible
    # here even though update_agent writes it; a marketplace binding makes the
    # gap worse (it silently overrides provider/model AND names the paying
    # wallet), so both are reported.
    if defn.provider:
        result["provider"] = defn.provider
    if defn.service_provider:
        # {provider_address, spec_digest} — plus who pays, which is the fact a
        # parent actually needs when reading this back.
        result["service_provider"] = dict(defn.service_provider)
        result["inference_payer"] = "self" if defn.identity else "unregistered"
    if defn.is_cognitive:
        result["agent_type"] = defn.agent_type
        result["max_turns"] = defn.max_turns
        result["tools"] = defn.tools
        if defn.parent_id:
            result["parent_id"] = defn.parent_id
        if defn.created_by:
            result["created_by"] = defn.created_by
        if getattr(defn, "cloned_from", None):
            result["cloned_from"] = defn.cloned_from
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
    # On-chain identity
    if defn.identity:
        result["agent_address"] = defn.identity.address
        result["lineage_hash"] = defn.identity.lineage_hash
        result["registration_tx"] = defn.identity.registration_tx
        # Live on-chain check if not already known to be registered
        if not defn.identity.registered_on_chain:
            try:
                from ..on_chain import OnChainService
                svc = OnChainService(runtime._config.rpb)
                if svc.available:
                    # Phase 12: agents are always identified by their own
                    # daemon-held keypair, never by the connected wallet.
                    # Check only the agent's own address.
                    if defn.identity.address and await svc.is_registered(defn.identity.address):
                        defn.identity.registered_on_chain = True
                        runtime.registry.persist_identity(defn.id)
            except Exception:
                pass
        result["registered_on_chain"] = defn.identity.registered_on_chain
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
    provider_or_model_changed = False
    if "model" in input or "provider" in input:
        # Same doctrine as service_provider below (2026-07-26,
        # employer-chooses-the-tool): which substrate an agent thinks on
        # is its PARENT's call. These two fields predate the gate and
        # were settable by any agent on any agent — including itself.
        caller_id = input.get("_caller_id")
        from . import is_owner_caller
        if caller_id is not None and not is_owner_caller(caller_id):
            if caller_id == agent_id:
                return {"error": "An agent cannot change its own model or "
                                 "provider — its substrate is its parent's "
                                 "call (docs/services_market.md, 2026-07-26)."}
            if defn.parent_id != caller_id:
                return {"error": "model/provider are parent-settable only: "
                                 f"'{caller_id}' is not the parent of "
                                 f"'{agent_id}'."}
    if "model" in input:
        defn.cognitive_model = input["model"]
        defn.provider = input["model"]
        changed.append("model")
        provider_or_model_changed = True
    if "provider" in input:
        # Explicit provider override (e.g. "rpb" for a dependent agent that
        # routes inference to a sponsor). Set after "model" so it wins when
        # both are present.
        defn.provider = input["provider"]
        changed.append("provider")
        provider_or_model_changed = True
    # NOTE: no per-agent "sponsor_address" here. Sponsored inference is a
    # DAEMON-level setting keyed on the owner wallet (ratified 2026-07-25,
    # docs/sponsored_inference.md) — configured once in the RPB Network
    # provider, not per agent. AgentDefinition.sponsor_address survives as a
    # dead field so old agent.yaml files still load; nothing reads it.
    if "service_provider" in input:
        # Marketplace inference binding (docs/services_market.md, ratified
        # 2026-07-26: employer-chooses-the-tool). An agent's MODEL/PROVIDER is
        # set only by its parent — an agent must never switch its own
        # substrate, so there is no self-set surface at all. A parent MAY buy a
        # marketplace inference service and provision a CHILD bound to it,
        # then scrutinize that child's output from outside the purchased
        # substrate; the child pays each call from its own funded wallet.
        #
        # Same gate shape as `budgets` above (parent-granted, no self-raise),
        # and it must be spelled out here: _update_agent has NO blanket
        # lineage gate, so an unguarded field is settable by any agent on any
        # agent — which on this field would let an agent redirect its own
        # cognition to a substrate of its choosing.
        caller_id = input.get("_caller_id")
        from . import is_owner_caller
        if caller_id is not None and not is_owner_caller(caller_id):
            if caller_id == agent_id:
                return {"error": "An agent cannot set or change its own "
                                 "service_provider binding — which substrate "
                                 "an agent thinks on is its parent's call "
                                 "(docs/services_market.md, 2026-07-26)."}
            if defn.parent_id != caller_id:
                return {"error": "service_provider is parent-settable only: "
                                 f"'{caller_id}' is not the parent of "
                                 f"'{agent_id}'."}
        raw_binding = input["service_provider"]
        if raw_binding is None:
            defn.service_provider = None
        else:
            from ..models import normalize_service_binding
            try:
                defn.service_provider = normalize_service_binding(raw_binding)
            except ValueError as exc:
                return {"error": str(exc)}
        changed.append("service_provider")
        # A binding change IS a provider change: it decides both which
        # substrate the agent thinks on and whose wallet pays, so the cached
        # provider instance must be evicted (see the eviction block below).
        provider_or_model_changed = True
    if "notify_parent" in input:
        defn.notify_parent = input["notify_parent"]
        changed.append("notify_parent")
    if "wake_parent_on_child" in input:
        defn.wake_parent_on_child = bool(input["wake_parent_on_child"])
        changed.append("wake_parent_on_child")
    if "concurrency" in input:
        defn.concurrency = input["concurrency"]
        changed.append("concurrency")
    if "budgets" in input:
        # Blessed semantics (docs/tool_substrate.md, reference default 3):
        # budgets are PARENT-updateable within the parent's own headroom.
        # Owner callers are unconstrained. An agent may change budgets
        # only for its direct children — never its own (no self-raise),
        # never a stranger's — and the cascade re-checks the new limits.
        caller_id = input.get("_caller_id")
        from . import is_owner_caller
        if caller_id is not None and not is_owner_caller(caller_id):
            if caller_id == agent_id:
                return {"error": "An agent cannot change its own budget — "
                                 "budgets are parent-granted."}
            if defn.parent_id != caller_id:
                return {"error": "Budgets are parent-updateable only: "
                                 f"'{caller_id}' is not the parent of "
                                 f"'{agent_id}'."}
        cascade_err = runtime.registry.validate_budget_update(
            defn, input["budgets"])
        if cascade_err:
            return {"error": cascade_err}
        defn.budgets = input["budgets"]
        changed.append("budgets")
    if "parent_id" in input:
        # Reparenting (the "regional manager" path): place a parent over a
        # formerly top-level agent, or re-home a subtree. Access control:
        # the owner is unconstrained; an agent may reparent only its OWN
        # direct child (delegating it further down) or adopt an agent under
        # ITSELF (become the new parent). It may never reparent a stranger
        # or move itself. The registry validates cycles / spawn+depth limits /
        # budget cascade and rolls back on any violation.
        caller_id = input.get("_caller_id")
        new_parent = input["parent_id"] or None
        from . import is_owner_caller
        if caller_id is not None and not is_owner_caller(caller_id):
            if caller_id == agent_id:
                return {"error": "An agent cannot reparent itself."}
            is_current_parent = defn.parent_id == caller_id
            is_new_parent = new_parent == caller_id
            if not (is_current_parent or is_new_parent):
                return {"error": "Reparenting is limited to the owner, the "
                                 "agent's current parent, or the adopting "
                                 "new parent."}
        err = runtime.registry.reparent_agent(agent_id, new_parent)
        if err:
            return {"error": err}
        changed.append("parent_id")

    if "connector_ids" in input:
        defn.connector_ids = input["connector_ids"]
        changed.append("connector_ids")
    if "secrets_allowance" in input:
        # The AUTHORED wish only (resolve_spec form: none/all/bundle/literal).
        # Safe to edit from any caller the subtree gate already admitted: the
        # ENFORCED grant is still the daemon-side intersection with the
        # parent's allowance at spawn (worker_host monotone clamp), so a wider
        # wish can never widen the actual grant. Takes effect at next spawn —
        # a live worker keeps the session it was minted.
        raw = input["secrets_allowance"]
        defn.secrets_allowance = (str(raw).strip() or None) if raw is not None else None
        changed.append("secrets_allowance")
    if "tools" in input:
        defn.tools = input["tools"]
        changed.append("tools")
    if "expose_as_tool" in input:
        defn.expose_as_tool = input["expose_as_tool"]
        changed.append("expose_as_tool")
    if "tool_input_schema" in input:
        defn.tool_input_schema = input["tool_input_schema"]
        changed.append("tool_input_schema")
    if "steps" in input:
        new_steps = []
        for s in input["steps"]:
            try:
                step_type = StepType(s["type"])
            except (ValueError, KeyError) as e:
                return {"error": f"Invalid step: {e}"}
            new_steps.append(StepDefinition(
                type=step_type,
                config=s.get("config", {}),
                name=s.get("name"),
            ))
        defn.steps = new_steps
        changed.append("steps")

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

    # §10: a provider/model change must be LIVE. The resolved provider instance
    # is cached in ``providers._active_providers[agent_id]``; without eviction
    # the cached instance keeps serving the OLD model (execution_engine.py:847
    # class of bug — inert update_agent). Evict + close so the next run rebuilds
    # against the new config. Mirrors unregister_agent's provider cleanup
    # (runtime/__init__.py:912-918) minus the file deletion.
    if provider_or_model_changed:
        pmgr = getattr(runtime, "providers", None)
        active = getattr(pmgr, "_active_providers", None) if pmgr else None
        if isinstance(active, dict):
            old_provider = active.pop(agent_id, None)
            if old_provider is not None and hasattr(old_provider, "close"):
                try:
                    await old_provider.close()
                except Exception as exc:
                    log.warning("Failed to close evicted provider for %s: %s", agent_id, exc)
            if hasattr(pmgr, "_cached_session_stats"):
                pmgr._cached_session_stats.pop(agent_id, None)

    # Persist to YAML
    try:
        save_agent(defn, runtime._config.agents_dir)
    except Exception as exc:
        log.warning("Agent updated in memory but YAML save failed: %s", exc)

    return {"agent_id": agent_id, "status": "updated", "changed": changed}


async def _create_agent(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    log.info("create_agent input: %s", json.dumps(input, default=str)[:2000])

    # Default to cognitive — matches the tool schema's advertised default and
    # the common case ("most agents are cognitive — just id, name, prompt").
    # The server previously defaulted to pipeline, contradicting the schema, so
    # an agent that omitted mode hit "must have at least one step".
    # EXCEPTION: explicit steps with no mode unambiguously describe a
    # pipeline — defaulting those to cognitive would silently DISCARD
    # the steps (callers predating the cognitive default do exactly this).
    default_mode = "pipeline" if input.get("steps") else "cognitive"
    mode_str = input.get("mode", default_mode)
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

        # System prompt: keep ONLY a caller-provided custom prompt. Do NOT
        # pre-render the delegate template here — leaving it empty lets the
        # execution engine build the (identity-free, byte-identical) cacheable
        # prompt fresh each run and deliver identity via the first user message.
        # Pre-rendering froze a per-agent prompt into the YAML, which defeated
        # cross-agent prefix caching (every agent re-created the ~50k prefix).
        system_prompt = input.get("system_prompt", "")

        effective_model = model or runtime._config.orchestrator.model or "sonnet"
        model_tier = get_model_tier(effective_model)
        # §10: an explicit provider (enum of known providers) pins routing.
        # Omitted → provider is the model id, and _resolve_provider_for_model
        # fails loud if it can't place it (no silent bridge fallback).
        explicit_provider = input.get("provider", "")

        # Marketplace inference binding (docs/services_market.md, ratified
        # 2026-07-26: employer-chooses-the-tool). No authority check is needed
        # HERE: parent_id is derived from caller_id above, so anything created
        # through this path is by construction the caller's own child. The
        # self-set case is impossible — an agent cannot create itself.
        service_binding = None
        if input.get("service_provider") is not None:
            from ..models import normalize_service_binding
            try:
                service_binding = normalize_service_binding(
                    input["service_provider"])
            except ValueError as exc:
                return {"error": str(exc)}

        # Warn if model tier seems low for agent_type
        tier_warning = None
        if agent_type in ("general",) and model_tier < 3:
            tier_warning = (
                f"Model '{effective_model}' is tier {model_tier} ({get_tier_label(model_tier)}). "
                f"Agent type '{agent_type}' works best with tier 3+ (autonomous) models."
            )
            log.warning("Low capability tier for agent: %s", tier_warning)

        defn = AgentDefinition(
            id=agent_id,
            name=input.get("name", "") or input.get("id", agent_id),
            mode=AgentMode.COGNITIVE,
            provider=explicit_provider or effective_model,
            cognitive_model=effective_model,
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
            notify_parent=input.get("notify_parent", True),
            secrets_allowance=input.get("secrets_allowance"),
            service_provider=service_binding,
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

                result = {"agent_id": aid, "status": "running", "execution_id": eid,
                         "capability_tier": model_tier, "tier_label": get_tier_label(model_tier)}
                if tier_warning:
                    result["tier_warning"] = tier_warning
                if service_binding:
                    result["service_provider"] = service_binding
                return result

            result = {"agent_id": aid, "status": "registered",
                      "capability_tier": model_tier, "tier_label": get_tier_label(model_tier)}
            if tier_warning:
                result["tier_warning"] = tier_warning
            if service_binding:
                result["service_provider"] = service_binding
            return result
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
            notify_parent=input.get("notify_parent", True),
            secrets_allowance=input.get("secrets_allowance"),
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
    # Legacy mode only: the auto-provisioned root agent is protected. In a
    # rootless fleet the orchestrator id is a normal agent (mirrors the
    # Runtime facade + registry guards).
    _orch_cfg = getattr(runtime._config, "orchestrator", None)
    if agent_id == ORCHESTRATOR_ID and getattr(_orch_cfg, "enabled", False):
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
    from ..orchestrator import ORCHESTRATOR_ID

    target = input["target"]
    defn = runtime.get_agent(target)
    if defn is None:
        return {"error": f"Agent '{target}' not found."}

    # Hierarchy scoping: agents can only message parent, direct children,
    # or siblings (same parent_id).  The owner surface is unrestricted.
    from . import is_owner_caller
    caller_id = input.get("_caller_id")
    if caller_id and not is_owner_caller(caller_id):
        caller_defn = runtime.get_agent(caller_id)
        if caller_defn:
            is_parent = (caller_defn.parent_id == target)
            is_child = (defn.parent_id == caller_id)
            is_sibling = (
                caller_defn.parent_id
                and defn.parent_id
                and caller_defn.parent_id == defn.parent_id
            )
            is_self = (caller_id == target)
            if not (is_parent or is_child or is_sibling or is_self):
                return {
                    "error": (
                        f"Agent '{caller_id}' cannot message '{target}': "
                        f"not a parent, child, or sibling. "
                        f"Agents can only message within their immediate hierarchy."
                    )
                }

    msg_type = MessageType(input.get("message_type") or input.get("type", "trigger"))
    priority = MessagePriority(input.get("priority", "normal"))
    instruction = (input.get("data") or {}).get("instruction", "")

    # NOTE: Do NOT add_user_turn here.  The execution engine records the
    # user turn right before calling the LLM (execution_engine.py lines
    # 624-636), which is the single place responsible for persisting turns
    # for both the orchestrator and child agents.  Adding it here caused
    # duplicate user messages in the conversation store.

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
    status = runtime.get_status(target)
    execution_id = None
    if defn.mode == AgentMode.COGNITIVE and status in (
        AgentStatus.ACTIVE, AgentStatus.COMPLETED, AgentStatus.ERROR
    ):
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
            return {"error": f"Invalid category: '{category_str}'. Use 'connector', 'pipeline', 'core', or 'registered'."}

    include_ops = bool(input.get("include_operations", False))
    tools = runtime.tool_registry.list_all(
        category=category,
        include_operations=include_ops,
        caller_id=input.get("_caller_id"),
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


async def _register_tool(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Author a tool manifest on the tool substrate (docs/tool_substrate.md).

    The author is the CALLER — authorship is the scoping primitive, so it
    is derived, never accepted as input. Owner callers author as "user".
    """
    from ..tool_store import OWNER_AUTHOR
    from . import is_owner_caller

    name = str(input.get("name") or "").strip()
    if not name:
        return {"error": "Missing required field: 'name'"}
    schema = input.get("input_schema")
    if not isinstance(schema, dict):
        return {"error": "input_schema must be a JSON-schema object"}

    # Registered tools may never shadow framework names.
    if get_core_tool_def(name) is not None:
        return {"error": f"'{name}' is a core ATN tool name; pick another"}
    for reserved in ("reg_", "pipeline_", "tool_", "connector_"):
        if name.startswith(reserved):
            return {"error": f"tool names may not start with '{reserved}'"}

    if input.get("endpoint"):
        return {"error": "endpoint-backed offerings are Services, not tools "
                         "— use the services rail (docs/services_market.md)"}

    caller_id = input.get("_caller_id")
    owner_caller = is_owner_caller(caller_id)
    # Publishing is a SEPARATE, case-by-case granted capability
    # (publish_tool) — register_tool never publishes for agents. Owner
    # callers may still register-and-publish in one step.
    if input.get("publish") and not owner_caller:
        return {"error": "register_tool no longer publishes: registration is "
                         "always private; publishing is the separate "
                         "publish_tool capability (granted case-by-case)"}
    author = OWNER_AUTHOR if owner_caller else str(caller_id)

    raw_deps = input.get("dependencies")
    dependencies = None
    if raw_deps is not None:
        if not isinstance(raw_deps, list) or not all(
            isinstance(d, str) for d in raw_deps
        ):
            return {"error": "dependencies must be a list of manifest digests"}
        dependencies = raw_deps or None

    capabilities = input.get("capabilities")
    if capabilities is not None and not isinstance(capabilities, dict):
        return {"error": "capabilities must be an object "
                         "({net, fs, spawn, env})"}

    try:
        result = runtime.tool_store.register(
            name=name,
            description=str(input.get("description") or ""),
            input_schema=schema,
            author=author,
            code=str(input.get("code") or ""),
            provider=str(input.get("provider") or ""),
            connector_id=str(input.get("connector_id") or ""),
            version_of=input.get("version_of") or None,
            publish=bool(input.get("publish", False)),
            dependencies=dependencies,
            capabilities=capabilities or None,
        )
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}
    return {
        "digest": result["digest"],
        "name": name,
        "trust_class": result["manifest"]["trust_class"],
        "author": author,
        "published": result["published"],
        "unified_name": f"reg_{result['digest'][:12]}",
    }


async def _publish_tool(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Publish an authored tool to the substrate.

    The agent OWNS its tool (authorship is structural); publishing is a
    separately granted capability — the grant IS the gate, case-by-case
    per agent, no approval queue. Author-only: you publish your own
    work, nobody else's.
    """
    from . import is_owner_caller

    digest = str(input.get("digest") or "")
    if not digest:
        return {"error": "Missing required field: 'digest'"}
    record = runtime.tool_store.resolve(digest)
    if record is None:
        return {"error": f"Unknown tool: {digest[:16]}"}

    caller_id = input.get("_caller_id")
    if record.origin == "adopted":
        return {"error": "adopted tools cannot be re-published — the "
                         "original author's publication stands"}
    if caller_id is not None and not is_owner_caller(caller_id):
        if record.author_id != caller_id:
            return {"error": "You can only publish tools you authored."}

    if record.published:
        return {"digest": record.digest, "name": record.name,
                "published": True, "note": "already published"}
    try:
        runtime.tool_store.set_published(record.digest, True)
    except ValueError as exc:  # orphan-author publish gate
        return {"error": str(exc)}
    return {"digest": record.digest, "name": record.name, "published": True}


async def _adopt_tool(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Adoption rail, agent half (docs/tool_substrate.md — Adoption).

    Proposes only — installation is the owner's WS-surface decision
    (approve_adoption). The caller is derived, never accepted as input.
    """
    from ..tool_store import OWNER_AUTHOR
    from . import is_owner_caller

    digest = str(input.get("digest") or "").strip()
    if not digest:
        return {"error": "Missing required field: 'digest'"}
    reason = str(input.get("reason") or "").strip()
    if not reason:
        return {"error": "Missing required field: 'reason' — the owner "
                         "decides on your justification"}
    caller_id = input.get("_caller_id")
    caller = OWNER_AUTHOR if is_owner_caller(caller_id) else str(caller_id)
    try:
        return await runtime.tool_store.propose_adoption(
            caller, digest, reason=reason)
    except Exception as exc:
        return {"error": str(exc)}


async def _vet_tool(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Inspection-review rail (v4.1 gradient trust — memory
    tool-economy-v4-gradient-trust).

    Inspect (no verdict) → manifest + pinned code; attest (verdict +
    report + optional per-axis scores) → local inspection row +
    consensus tool_used event with vet=True. The inspection moves the
    tool's public position/rating (weighted by the caller's reputation
    × credibility) and mints NOTHING. The caller is derived, never
    accepted as input.
    """
    from ..tool_store import OWNER_AUTHOR
    from . import is_owner_caller

    digest = str(input.get("digest") or "").strip()
    if not digest:
        return {"error": "Missing required field: 'digest'"}
    caller_id = input.get("_caller_id")
    caller = OWNER_AUTHOR if is_owner_caller(caller_id) else str(caller_id)
    verdict = input.get("verdict")
    raw_axes = input.get("axes")
    axes = raw_axes if isinstance(raw_axes, dict) else None
    try:
        return await runtime.tool_store.vet_tool(
            caller, digest,
            verdict=str(verdict) if verdict is not None else None,
            report=str(input.get("report") or ""),
            axes=axes,
        )
    except Exception as exc:
        return {"error": str(exc)}


async def _check_evidence(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Evidence-replay verify-then-support (docs/tool_substrate.md — Evidence).

    Replays a CON's reproducible failing invocation against the pinned
    tool locally and, on confirmation, posts a support sprout under the
    CON. Caller is derived; the replay itself is owner-sanctioned local
    verification, so scoping is not the replay's concern.
    """
    digest = str(input.get("manifest_digest") or "").strip()
    if not digest:
        return {"error": "Missing required field: 'manifest_digest'"}
    evidence = input.get("evidence")
    if not isinstance(evidence, dict):
        return {"error": "'evidence' must be an object "
                         "(args_json + expected_error/expected_digest)"}
    con_node_id = str(input.get("con_node_id") or "").strip()
    support = input.get("support")
    support = True if support is None else bool(support)
    caller_id = input.get("_caller_id")
    try:
        return await runtime.tool_store.check_evidence(
            caller_id, digest, evidence,
            con_node_id=con_node_id, support=support,
        )
    except Exception as exc:
        return {"error": str(exc)}


async def _run_trial(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Service verifier trial (docs/tool_substrate.md — Verifier trials;
    venture_vault_design memory).

    Fetches a venture prospectus, executes its declared black-box trial
    battery against the service's MCP surface, scores pass/fail per the
    prospectus's pre-committed criteria, blob-stores a trial report, and
    returns the verdict + report digest + attestTrial calldata so the
    owner surface can submit the on-chain trial record. Caller derived.
    """
    from ..tool_store import OWNER_AUTHOR
    from . import is_owner_caller

    prospectus_digest = str(input.get("prospectus_digest") or "").strip()
    if not prospectus_digest:
        return {"error": "Missing required field: 'prospectus_digest'"}
    caller_id = input.get("_caller_id")
    caller = OWNER_AUTHOR if is_owner_caller(caller_id) else str(caller_id)
    try:
        return await runtime.trial_runner.run_trial(caller, prospectus_digest)
    except Exception as exc:
        return {"error": str(exc)}


async def _attest_tools(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Record cognitive attestations for a closed work item.

    The caller is derived (authorship/scoping primitive); owner callers
    attest as "user". This is the per-work-item reflection step — the only
    usage that counts toward a tool author's mint (docs/tool_substrate.md).
    """
    from ..tool_store import OWNER_AUTHOR
    from . import is_owner_caller

    judgments = input.get("judgments")
    if not isinstance(judgments, list):
        return {"error": "judgments must be an array"}
    context = str(input.get("context") or "")
    if not context:
        return {"error": "Missing required field: 'context'"}

    caller_id = input.get("_caller_id")
    caller = OWNER_AUTHOR if is_owner_caller(caller_id) else str(caller_id)

    try:
        return runtime.tool_store.attest_usage(caller, judgments, context)
    except Exception as exc:
        return {"error": str(exc)}


async def _probe_tools(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Agent-facing library search (v3): semantic retrieval over published
    tool manifests, ranked by review-drifted ratings. Falls back to a
    substring match over the local store when the substrate is down."""
    query = str(input.get("query") or "").strip()
    if not query:
        return {"error": "Missing required field: 'query'"}
    try:
        k = max(1, min(int(input.get("k") or 8), 25))
    except (TypeError, ValueError):
        k = 8

    matches: list[dict[str, Any]] = []
    source = "local"
    try:
        autonet = getattr(runtime, "autonet", None)
        service = getattr(autonet, "_service", None) if autonet else None
        world_service = getattr(service, "_world_service", None) if service else None
        if world_service is not None:
            from nodes.common.world_model_substrate.tool_manifest import (
                is_tool_manifest,
            )
            # v4.1 trust picture: join review mass + inspection count from
            # the economy graph so each match carries how much earned voice
            # is behind its score (same close state the tool_reviews /
            # economy_graph surfaces read — no new close read invented).
            eg = _economy_graph_snapshot(world_service)
            positions = eg.get("positions", {})
            vetting = eg.get("vetting", {})
            result = world_service.infer_artifacts(query, k=max(k * 3, 12))
            for art in result.get("artifacts", []):
                payload = art.get("payload")
                if not is_tool_manifest(payload):
                    continue
                digest = art.get("digest", "")
                matches.append(_trust_row(
                    digest, payload,
                    score=art.get("final", 0.0),
                    rating=art.get("rating", 0.0),
                    axes=art.get("axes", []),
                    positions=positions, vetting=vetting,
                    tool_store=runtime.tool_store,
                ))
                if len(matches) >= k:
                    break
            source = "substrate"
    except Exception as exc:
        log.debug("probe_tools substrate path failed: %s", exc)
    if not matches:
        needle = query.lower()
        for record in runtime.tool_store.visible_to(None):
            hay = f"{record.name} {record.manifest.get('description', '')}".lower()
            if all(w in hay for w in needle.split()):
                matches.append({
                    "digest": record.digest,
                    "name": record.name,
                    "description": record.manifest.get("description", ""),
                    "author": record.author,
                    "trust_class": record.trust_class,
                    "score": 0.0,
                    "rating": 0.0,
                    "axes": [],
                    "mass": [],
                    "review_mass": 0.0,
                    "inspections": 0,
                })
                if len(matches) >= k:
                    break
        source = "local"
    return {"matches": matches, "source": source}


def _economy_graph_snapshot(world_service: Any) -> dict[str, Any]:
    """Best-effort economy-graph read for the probe trust picture; an
    empty dict on any failure keeps probe_tools working without it."""
    try:
        return world_service.read_economy_graph(last_n_epochs=10) or {}
    except Exception as exc:  # noqa: BLE001 — trust picture is additive
        log.debug("probe_tools economy graph read failed: %s", exc)
        return {}


def _trust_row(
    digest: str,
    payload: dict[str, Any],
    *,
    score: Any,
    rating: Any,
    axes: Any,
    positions: dict[str, Any],
    vetting: dict[str, Any],
    tool_store: Any,
) -> dict[str, Any]:
    """One probe_tools match enriched with the v4.1 trust picture:
    review mass (how much earned voice is behind the score), inspection
    count, and author household when resolvable locally."""
    pos = positions.get(digest) or {}
    mass = [float(x) for x in (pos.get("mass") or [])]
    vet = vetting.get(digest) or {}
    author = payload.get("author", "")
    row: dict[str, Any] = {
        "digest": digest,
        "name": payload.get("name", ""),
        "description": payload.get("description", ""),
        "author": author,
        "trust_class": payload.get("trust_class", ""),
        "score": score,
        "rating": rating,
        "axes": axes,
        # Trust picture (v4.1): mass = per-axis review weight behind the
        # drifted score; review_mass = its peak (zero = nobody has
        # reviewed — inspect before trusting); inspections = code reads.
        "mass": mass,
        "review_mass": max(mass) if mass else 0.0,
        "inspections": int(vet.get("validators") or 0),
    }
    household = _resolve_household(tool_store, author)
    if household:
        row["author_household"] = household
    return row


def _resolve_household(tool_store: Any, author: str) -> str:
    """Author household from local state when available; empty otherwise.
    Households are close-side aggregation — locally we can only surface
    the author's own address, so this stays best-effort and never fails
    the probe."""
    return ""


# ---------------------------------------------------------------------------
# Services rail (docs/services_market.md) — find/register/pay/request
# ---------------------------------------------------------------------------

def _caller_identity(runtime: Runtime, caller_id: str | None):
    """Resolve the calling agent's on-chain identity + stored key.

    Returns (defn, identity, private_key) or an ``{"error": ...}`` dict when
    the caller isn't a registered agent with a daemon-held key. Owner callers
    have no agent key of their own — a service author/payer must be an agent.
    """
    from . import is_owner_caller

    if caller_id is None or is_owner_caller(caller_id):
        return {"error": "This tool must be called by a registered agent — the "
                         "owner surface has no agent key to sign with. Ask an "
                         "agent (or register one) to author/pay for services."}
    defn = runtime.get_agent(caller_id)
    if defn is None:
        return {"error": f"Agent '{caller_id}' not found."}
    if not defn.identity or not defn.identity.address:
        return {"error": f"Agent '{caller_id}' has no on-chain identity — "
                         "register it first (register_on_chain)."}
    key = runtime.registry.get_agent_key(caller_id)
    if not key:
        return {"error": f"No private key stored for agent '{caller_id}'."}
    return defn, defn.identity, key


async def _find_services(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """List on-chain services (most-recent first), optionally substring-filtered
    against locally-known spec metadata."""
    query = str(input.get("query") or "").strip().lower()
    try:
        limit = max(1, min(int(input.get("limit") or 25), 200))
    except (TypeError, ValueError):
        limit = 25

    from ..on_chain import ServiceMarketClient
    smc = ServiceMarketClient(runtime._config.rpb)
    if not smc.registry_available:
        return {"error": "ServiceRegistry not configured "
                         "(missing service_registry_address or rpc_url)."}

    services = await smc.list_services()
    # Most-recent first (list_services returns ascending serviceId).
    services = list(reversed(services))

    store = getattr(runtime, "service_store", None)
    rows: list[dict[str, Any]] = []
    for s in services:
        digest = str(s.get("spec_digest") or "")
        name = ""
        description = ""
        # Enrich with locally-known metadata + drive the substring filter.
        rec = store.get(digest) if store is not None else None
        if rec is not None:
            name = rec.name
            description = str(rec.spec.get("description") or "")
        if query:
            hay = f"{name} {description} {digest} {s.get('provider','')}".lower()
            if query not in hay:
                continue
        rows.append({
            "service_id": s.get("service_id"),
            "provider": s.get("provider"),
            "ask": s.get("ask_amount"),
            "spec_digest": digest,
            "active": s.get("active"),
            "name": name,
            "description": description,
        })
        if len(rows) >= limit:
            break
    return {"services": rows, "count": len(rows)}


async def _register_service(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Author a service: persist the local spec (service_store) AND register it
    on-chain in the ServiceRegistry under the caller's agent key."""
    caller_id = input.get("_caller_id")
    resolved = _caller_identity(runtime, caller_id)
    if isinstance(resolved, dict):
        return resolved
    _defn, identity, key = resolved

    name = str(input.get("name") or "").strip()
    if not name:
        return {"error": "Missing required field: 'name'"}
    description = str(input.get("description") or "").strip()
    if not description:
        return {"error": "Missing required field: 'description'"}
    ask_amount = input.get("ask_amount")
    try:
        ask_amount = int(ask_amount)
    except (TypeError, ValueError):
        return {"error": "ask_amount must be an integer (ATN base units)"}
    if ask_amount < 0:
        return {"error": "ask_amount must be non-negative"}

    schema = input.get("input_schema")
    if schema is not None and not isinstance(schema, dict):
        return {"error": "input_schema must be a JSON-schema object"}
    schema = schema or {"type": "object", "properties": {}}
    output_schema = input.get("output_schema")
    if output_schema is not None and not isinstance(output_schema, dict):
        return {"error": "output_schema must be a JSON-schema object or absent"}

    store = getattr(runtime, "service_store", None)
    if store is None:
        return {"error": "Service store not available on this daemon."}

    # Asks are ATN-denominated by construction (ratified 2026-07-10) —
    # no token field.
    ask = {"amount": str(ask_amount), "unit": "per_item"}
    try:
        built = store.register(
            name=name,
            description=description,
            input_schema=schema,
            author=str(caller_id),
            ask=ask,
            backing_tool=str(input.get("backing_tool") or ""),
            output_schema=output_schema,
            endpoint_hint=str(input.get("endpoint_hint") or ""),
            image_uri=str(input.get("image_uri") or "").strip(),
        )
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}
    spec_digest = built["digest"]

    from ..on_chain import ServiceMarketClient
    smc = ServiceMarketClient(runtime._config.rpb)
    if not smc.registry_available:
        return {"error": "Local spec persisted but ServiceRegistry not "
                         "configured (missing service_registry_address / "
                         "rpc_url) — cannot register on-chain.",
                "spec_digest": spec_digest}

    result = await smc.register_service(key, spec_digest, ask_amount)
    if not result.get("success"):
        return {"error": f"On-chain registration failed: {result.get('error')}",
                "spec_digest": spec_digest}
    return {
        "service_id": result.get("service_id"),
        "tx_hash": result.get("tx_hash"),
        "spec_digest": spec_digest,
        "provider": identity.address,
    }


async def _pay_for_service(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Sign a Substrate.payForService transfer from the caller's key; returns
    {tx_hash, request_id} for the consumer to hand to the provider.

    Thin wrapper over ``atn.service_client.pay_for_service`` (shared with the
    ``service`` inference provider), which owns the chain mechanics."""
    from .. import service_client

    caller_id = input.get("_caller_id")
    resolved = _caller_identity(runtime, caller_id)
    if isinstance(resolved, dict):
        return resolved
    _defn, _identity, key = resolved

    recipient = str(input.get("recipient") or "").strip()
    if not recipient:
        return {"error": "Missing required field: 'recipient'"}
    if input.get("amount") is None:
        return {"error": "Missing required field: 'amount'"}

    return await service_client.pay_for_service(
        runtime._config.rpb,
        key,
        recipient,
        input.get("amount"),
        request_id=str(input.get("request_id") or "").strip(),
    )


async def _request_service(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Consumer cross-daemon call: resolve the provider's on-chain WS endpoint,
    connect one-shot, send a service_request with the payment proof, return the
    reply.

    Thin wrapper over ``atn.service_client.request_service`` (shared with the
    ``service`` inference provider)."""
    from .. import service_client

    provider_address = str(input.get("provider_address") or "").strip()
    service_id = str(input.get("service_id") or "").strip()
    payload = input.get("payload")
    tx_hash = str(input.get("tx_hash") or "").strip()
    request_id = str(input.get("request_id") or "").strip()
    if not provider_address:
        return {"error": "Missing required field: 'provider_address'"}
    if not service_id:
        return {"error": "Missing required field: 'service_id'"}
    if not isinstance(payload, dict):
        return {"error": "'payload' must be an object"}
    if not tx_hash or not request_id:
        return {"error": "Missing payment proof: both 'tx_hash' and "
                         "'request_id' are required (from pay_for_service)."}

    caller_id = input.get("_caller_id")
    defn = runtime.get_agent(caller_id) if caller_id else None
    client_addr = (defn.identity.address
                   if defn and defn.identity else "")

    return await service_client.request_service(
        runtime._config.rpb,
        provider_address,
        service_id,
        payload,
        tx_hash=tx_hash,
        request_id=request_id,
        client_address=client_addr,
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


async def _get_my_budget_status(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Return the calling agent's budget posture.

    Includes own caps, ancestors' caps + headroom, and subscription utilization
    for any provider whose name maps to a known subscription bridge.
    """
    caller_id = input.get("_caller_id") or ORCHESTRATOR_ID
    own = runtime.registry.get_budget_info(caller_id)

    # Uniform effective-limits rail: an agent WITHOUT a per-agent budget must
    # see its limits through the SAME shape/verbiage as a budgeted one — for it
    # the "budget" is the daemon-wide provider ceiling (dollar cap or inferred
    # subscription remaining). Route both cases through one function.
    from ..effective_limits import compute_effective_limits
    effective = compute_effective_limits(
        caller_id,
        registry=runtime.registry,
        metering=getattr(runtime, "metering", None),
        config=getattr(runtime, "_config", None),
    ).to_dict()

    # Walk ancestors: report each one's cap and remaining headroom for the
    # providers the caller has declared (or for all of theirs if caller has none).
    ancestors: list[dict[str, Any]] = []
    cur_defn = runtime.registry.get_agent(caller_id)
    seen: set[str] = {caller_id}
    while cur_defn and cur_defn.parent_id:
        parent_id = runtime.registry._resolve_parent_agent_id(cur_defn.parent_id)
        if parent_id in seen or parent_id not in runtime.registry._agents:
            break
        seen.add(parent_id)
        info = runtime.registry.get_budget_info(parent_id)
        if info:
            ancestors.append({
                "agent_id": parent_id,
                "budgets": info,
            })
        cur_defn = runtime.registry.get_agent(parent_id)

    # Subscription utilization (Claude Max, etc.). Only relevant if the agent
    # uses a subscription provider. We check the active providers map for a
    # provider whose `_rate_limits` is populated.
    subscription: dict[str, Any] = {}
    try:
        active = getattr(runtime, "_active_providers", None) or {}
        for prov in active.values():
            if hasattr(prov, "_rate_limits") and getattr(prov, "_rate_limits"):
                subscription[getattr(prov, "name", "unknown")] = {
                    "rate_limits": dict(prov._rate_limits),
                    "tokens_per_percent": getattr(prov, "tokens_per_percent", None),
                    "tokens_per_pct_by_class": getattr(
                        prov, "tokens_per_pct_by_class", {},
                    ),
                }
    except Exception:
        pass

    # P5: an isolated bridge agent's provider lives in the WORKER, not in
    # ``_active_providers``, so the live-provider scan above finds nothing for it.
    # The worker reports its subscription snapshot (rate_limits +
    # tokens_per_pct_by_class) on execution_done; WorkerHost caches it under the
    # agent_id in ``provider_manager._cached_session_stats``. Consult that cache
    # for the CALLER so get_my_budget_status answers for isolated bridge agents.
    # The snapshot is as-of the last completed orchestration (see WorkerHost).
    try:
        pmgr = getattr(runtime, "providers", None)
        cache = getattr(pmgr, "_cached_session_stats", None)
        cached = cache.get(caller_id) if isinstance(cache, dict) else None
        key = "claude_max"
        rl = cached.get("rate_limits") if isinstance(cached, dict) else None
        if rl and key not in subscription:
            subscription[key] = {
                "rate_limits": dict(rl),
                "tokens_per_percent": cached.get("tokens_per_percent"),
                "tokens_per_pct_by_class": cached.get(
                    "tokens_per_pct_by_class", {}),
                "source": "worker-snapshot",
            }
        elif not rl and key not in subscription:
            # Supervised bridge agent but no usable snapshot yet — either no
            # cache entry at all (common: first turn hasn't completed) or an
            # entry with empty rate_limits. Answer "known-unknown" rather than
            # silently empty so the caller can tell the difference. This must
            # fire even when ``cached is None``, so it lives OUTSIDE the
            # isinstance(cached, dict) guard.
            is_supervised = False
            sup = getattr(runtime, "supervisor", None)
            if sup is not None:
                try:
                    is_supervised = bool(sup.is_supervised(caller_id))
                except Exception:
                    is_supervised = False
            if is_supervised:
                subscription[key] = {
                    "status": "at-worker",
                    "rate_limits": {},
                    "source": "worker-snapshot-pending",
                }
    except Exception:
        pass

    return {
        "caller_id": caller_id,
        "own": own,
        "effective_limits": effective,
        "ancestors": ancestors,
        "subscription": subscription,
    }


# §13 get_usage: cache the (possibly network-backed) usage payload per caller
# so an agent polling get_usage every turn doesn't fire a probe call each time.
_USAGE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_USAGE_CACHE_TTL = 60.0


async def _get_usage(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Return the caller's provider usage (§13).

    Bridge (Claude Max) callers get the cached subscription-window utilization
    (5h/7d + reset times from ``BridgeProvider.refresh_usage``) plus their own
    per-provider budget counters. API/local callers get cumulative session
    tokens vs their configured budget. Cached ~60s per caller.
    """
    import time
    from . import ORCHESTRATOR_ID
    from ..providers.bridge import BridgeProvider

    caller_id = input.get("_caller_id") or ORCHESTRATOR_ID

    cached = _USAGE_CACHE.get(caller_id)
    if cached is not None and (time.monotonic() - cached[0]) < _USAGE_CACHE_TTL:
        return {**cached[1], "cached": True}

    result: dict[str, Any] = {"caller_id": caller_id}

    # The caller's own per-provider budget counters (works for every provider).
    try:
        result["budgets"] = runtime.registry.get_budget_info(caller_id)
    except Exception:
        result["budgets"] = {}

    pmgr = getattr(runtime, "providers", None)
    active = getattr(pmgr, "_active_providers", None) if pmgr else None
    provider = active.get(caller_id) if isinstance(active, dict) else None

    if isinstance(provider, BridgeProvider):
        # Subscription-window utilization. Prefer the already-cached
        # rate_limits; only hit the network if we've never populated them.
        result["provider"] = "claude_max"
        rate_limits = dict(getattr(provider, "_rate_limits", {}) or {})
        if not rate_limits:
            try:
                rate_limits = await provider.refresh_usage()
            except Exception as exc:
                log.warning("get_usage: refresh_usage failed for %s: %s", caller_id, exc)
                rate_limits = {}
        result["subscription"] = {"rate_limits": rate_limits}
        # Session token counters, if the bridge exposes them.
        try:
            stats = provider.session_stats
            result["session"] = {
                "cumulative_input_tokens": stats.get("cumulative_input_tokens"),
                "cumulative_output_tokens": stats.get("cumulative_output_tokens"),
                "context_used_pct": stats.get("context_used_pct"),
            }
        except Exception:
            pass
    else:
        # API / local provider: cumulative session tokens vs configured budget.
        if provider is not None and hasattr(provider, "session_stats"):
            try:
                stats = provider.session_stats
                result["provider"] = getattr(provider, "name", "unknown")
                result["session"] = stats
            except Exception:
                result["provider"] = getattr(provider, "name", "unknown")
        else:
            # No live provider (e.g. isolated worker) — fall back to the cached
            # session-stats snapshot the worker reported on execution_done.
            cache = getattr(pmgr, "_cached_session_stats", None)
            snap = cache.get(caller_id) if isinstance(cache, dict) else None
            if snap:
                result["session"] = snap
                if snap.get("rate_limits"):
                    result["subscription"] = {"rate_limits": snap["rate_limits"]}

    _USAGE_CACHE[caller_id] = (time.monotonic(), result)
    return {**result, "cached": False}


async def _metering_report(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Daemon-wide operational metering report for the admin agent.

    Thin wrapper over MeteringService.report — all math is deterministic in
    metering.py; this only shapes the args and handles the missing-service case.
    """
    metering = getattr(runtime, "metering", None)
    if metering is None:
        return {"error": "metering service unavailable on this daemon"}
    try:
        window_hours = float(input.get("window_hours", 24) or 24)
    except (TypeError, ValueError):
        window_hours = 24.0
    bucket = input.get("bucket", "hour")
    if bucket not in ("hour", "day"):
        bucket = "hour"
    try:
        return metering.report(window_hours=window_hours, bucket=bucket)
    except Exception as exc:
        log.warning("metering_report failed: %s", exc, exc_info=True)
        return {"error": f"metering report failed: {exc}"}


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
    "update_agent",       # update self or direct children
    "delegate_status",    # check sub-agent status with timestamps
    "delegate_collect",   # wait for sub-agent result
    "delegate_message",   # send message to running sub-agent
    "get_latest_thought", # lightweight check on agent activity
    "get_children_status",  # compact status for all direct children
    "trigger_run",    # trigger agent execution
    "get_output",     # read child agent output
    "post_message",   # communicate with other agents
    "get_snapshot",   # see system state
    "compact_agent",  # §15 compact a direct child's context
}


def _get_delegate_tools() -> list[dict[str, Any]]:
    """Build the tool list for a delegate sub-agent."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in _TOOLS
        if t.name in _DELEGATE_TOOL_NAMES
    ]


# ---------------------------------------------------------------------------
# Tool categories — granular tool selection for agent creation
# ---------------------------------------------------------------------------
# Categories map to tool names.  An agent's `tools` list can contain:
#   - Category names (e.g. "delegation", "messaging") → expanded to tool names
#   - Legacy flags: "atn_full" → all tools, "atn_core" → delegate subset
#   - "sdk_builtin" / "connectors" → handled by execution_engine, not here

_TOOL_CATEGORIES: dict[str, set[str]] = {
    "delegation": {
        "create_agent", "update_agent", "delegate_status", "delegate_collect",
        "delegate_message", "get_latest_thought", "get_children_status",
        "trigger_run", "get_output", "compact_agent",
    },
    "messaging": {"post_message"},
    "observation": {"get_snapshot", "list_agents", "get_agent", "get_execution", "get_history"},
    "lifecycle": {"activate_agent", "deactivate_agent", "kill_agent", "kill_execution",
                  "remove_agent", "compact_agent"},
    "connectors": {"list_connectors", "add_connector", "get_connector_tools", "use_connector", "remove_connector"},
    # Discovery (probe_tools) and the post-use review (attest_tools) ride
    # the same bundle as list/use (v3): every agent that can USE registered
    # tools must be able to search for them and review them — the review is
    # the signal that routes discovery, so a use-without-review surface
    # would be a broken grant.
    "unified_tools": {"list_tools", "use_tool", "probe_tools", "attest_tools"},
    # Tool substrate (docs/tool_substrate.md): authoring is opt-in per
    # agent. attest_tools also stays here for back-compat with existing
    # toolsmith-only grants.
    "toolsmith": {"register_tool", "attest_tools"},
    # Publishing is deliberately its OWN bundle (user, 2026-07-05):
    # whether an agent may publish its tools to the substrate is a
    # case-by-case grant — having toolsmith never implies it.
    "publishing": {"publish_tool"},
    # Vetting is the validator/inspection role (v4.1 gradient trust) —
    # its own case-by-case grant, same doctrine as publishing: reading
    # foreign code and moving a tool's public rating with your reputation
    # is not implied by authoring or publishing. It also carries the two
    # evidence-grade validator flows: check_evidence (replay an evidence-CON and back it
    # — Evidence section) and run_trial (probe a venture's service moat
    # against its prospectus battery — Verifier trials).
    "vetting": {"vet_tool", "check_evidence", "run_trial"},
    # Adoption (spec: Adoption rail) — the agent may PROPOSE installing
    # network tools; the owner approves per tool. Case-by-case grant:
    # publishing risks reputation, adoption risks the host.
    "adoption": {"adopt_tool"},
    "planning": {"get_goals", "add_goal", "update_goal", "get_projects", "add_project", "update_project",
                 "propose_task", "list_tasks"},
    "budget": {"get_credit_budget", "set_credit_budget", "get_usage"},
    # Ops/admin: daemon-wide metering view (cost series, cache anomalies,
    # per-agent burn, quota trajectory). Its own bundle — reading the whole
    # daemon's spend is a privileged, admin-only surface, not implied by an
    # agent knowing its own budget (get_usage / get_my_budget_status).
    "metering": {"metering_report"},
    "identity": {"register_on_chain"},
    # Services rail (docs/services_market.md): its own case-by-case bundle.
    # find_services is read-only discovery; register_service authors a paid
    # remote API under the agent's key; pay_for_service + request_service are
    # the consumer half (sign a payForService transfer, then cross-daemon
    # invoke with the proof). Kept off the progressive surface.
    "services": {"find_services", "register_service",
                 "pay_for_service", "request_service"},
    "profile": {"get_user_profile"},
    # "shell" (bash/read_file/write_file/list_directory/search_files) is NOT a
    # normal category: its tools live in shell_tools.py, not _TOOLS, and are
    # appended by execution_engine for non-bridge providers. resolve_tool_surface
    # skips it (like "sdk_builtin"). §9: non-bridge agents opt in via
    # tools=["shell", ...] instead of getting the whole schema block by default.
    "shell": set(),
}


# High-frequency tools always included in the progressive surface.
# Everything else is discoverable via list_tools/use_tool.
_PROGRESSIVE_DIRECT_TOOLS = {
    "create_agent", "update_agent",
    "post_message", "get_snapshot", "get_output",
    "delegate_status", "delegate_collect", "get_children_status",
    "list_tools", "use_tool",
}


def resolve_tool_surface(tool_spec: list[str]) -> list[dict[str, Any]]:
    """Resolve a tool spec (categories, flags, or tool names) to tool definitions.

    Returns a list of tool dicts ready for the bridge.
    """
    if not tool_spec:
        # Default: delegate subset
        return _get_delegate_tools()

    # Legacy flags
    if "atn_full" in tool_spec:
        return get_tool_definitions_for_bridge()

    # Progressive: high-frequency tools direct, rest via list_tools/use_tool
    if "atn_progressive" in tool_spec:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in _TOOLS
            if t.name in _PROGRESSIVE_DIRECT_TOOLS
        ]

    # Collect tool names from categories and explicit names
    resolved_names: set[str] = set()

    for spec in tool_spec:
        if spec in ("sdk_builtin", "connectors", "shell"):
            continue  # handled by execution_engine (shell → _SHELL_TOOLS, §9)
        if spec == "atn_core":
            resolved_names.update(_DELEGATE_TOOL_NAMES)
        elif spec in _TOOL_CATEGORIES:
            resolved_names.update(_TOOL_CATEGORIES[spec])
        else:
            # Treat as an explicit tool name
            resolved_names.add(spec)

    # If nothing resolved (e.g. only "sdk_builtin"), fall back to delegate set
    if not resolved_names:
        return _get_delegate_tools()

    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in _TOOLS
        if t.name in resolved_names
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

    # Try live injection (direct delivery). Dual-mode: for an in-process agent
    # this hits the daemon-resident provider; for an ISOLATED worker the
    # provider lives in the worker, so control.send_delegate_message routes the
    # injection over the IPC "send_user_message" cmd. Either way a True return
    # means the running loop received it.
    control = getattr(runtime, "control", None)
    if control is not None:
        try:
            if await control.send_delegate_message(agent_id, content):
                return {"status": "delivered", "agent_id": agent_id}
        except Exception:
            pass  # fall through to the inbox fallback below

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
        AgentStatus.BUDGET_PAUSED: "budget_paused",
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


async def _get_children_status(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Get compact status for all direct children of the calling agent."""
    from . import ORCHESTRATOR_ID

    caller_id = input.get("_caller_id") or ORCHESTRATOR_ID
    children = runtime.get_children(caller_id)

    if not children:
        return {"caller_id": caller_id, "children": [], "count": 0}

    data_dir = runtime._config.data_dir
    result_children: list[dict[str, Any]] = []

    for child in children:
        child_id = child.id
        status = runtime.get_status(child_id)

        entry: dict[str, Any] = {
            "id": child_id,
            "name": child.name,
            "status": status.value if status else "unknown",
        }

        # Get delegate registry info for turns and last_tool
        node = runtime.delegate_registry.get_node(child_id)
        if node:
            entry["turns"] = node.turns
            entry["last_tool"] = node.last_tool

        # Determine conversation_path based on agent mode
        if child.mode == AgentMode.COGNITIVE:
            # Cognitive agents have conversation JSONL
            conv_path = data_dir / "agents" / child_id / "conversation.jsonl"
        else:
            # Non-cognitive agents use delegate output log
            conv_path = data_dir / "delegates" / f"{child_id}.log"

        entry["conversation_path"] = str(conv_path)

        # Add budget info if available
        budget_info = runtime.registry.get_budget_info(child_id)
        if budget_info:
            entry["budget"] = budget_info

        result_children.append(entry)

    return {
        "caller_id": caller_id,
        "children": result_children,
        "count": len(result_children),
    }


async def _register_on_chain(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Register a child agent on-chain via the daemon's stored private key."""
    agent_id = input["agent_id"]
    sponsor_address = input.get("sponsor_address", "")

    defn = runtime.get_agent(agent_id)
    if defn is None:
        return {"error": f"Agent '{agent_id}' not found."}
    if not defn.identity:
        return {"error": f"Agent '{agent_id}' has no identity (no system prompt?)."}
    if defn.identity.registered_on_chain:
        return {"error": f"Agent '{agent_id}' is already registered on-chain.",
                "agent_address": defn.identity.address}

    private_key = runtime.registry.get_agent_key(agent_id)
    if not private_key:
        return {"error": f"No private key stored for '{agent_id}'. "
                         "Root agents must register via the frontend wallet."}

    from ..on_chain import OnChainService
    svc = OnChainService(runtime._config.rpb)
    if not svc.available:
        return {"error": "On-chain service not configured (missing rpb_contract_address or rpc_url)."}

    parent_address = ""
    if defn.parent_id:
        parent_defn = runtime.get_agent(defn.parent_id)
        if parent_defn and parent_defn.identity:
            parent_address = parent_defn.identity.address

    result = await svc.register_agent(
        identity=defn.identity,
        private_key=private_key,
        system_prompt=defn.system_prompt or "",
        parent_address=parent_address,
        sponsor_address=sponsor_address,
    )

    if result.get("success"):
        defn.identity.registered_on_chain = True
        defn.identity.registration_tx = result.get("tx_hash", "")
        log.info("Agent %s registered on-chain: tx=%s", agent_id, result.get("tx_hash"))

    return result


# ---------------------------------------------------------------------------
# Manual compaction (§15)
# ---------------------------------------------------------------------------

async def _compact_agent(runtime: Runtime, input: dict[str, Any]) -> dict[str, Any]:
    """Compact a target agent's conversation to free context (§15).

    Permissions: the owner (ORCHESTRATOR_ID caller — this includes the WS/owner
    surface, whose caller_id defaults to the orchestrator) may compact any
    agent; a non-owner agent may compact only its DIRECT CHILDREN
    (target.parent_id == caller_id); no agent may compact itself. Violations
    return an error with no side effects.

    Dispatch by target state:
      - running generic loop  -> set the provider's compact-requested flag,
        honored at the next iteration boundary (status "queued").
      - running bridge (SDK)  -> no input compact control -> "unsupported_while_running".
      - idle (any provider)   -> summarize the persisted store in place, archive
        the original, evict the cached provider (status "compacted").
    """
    from . import ORCHESTRATOR_ID

    agent_id = input.get("agent_id", "")
    if not agent_id:
        return {"error": "Missing 'agent_id'."}
    caller_id = input.get("_caller_id") or ORCHESTRATOR_ID

    target = runtime.get_agent(agent_id)
    if target is None:
        return {"error": f"Agent '{agent_id}' not found."}

    # --- Permission check (§15) ---------------------------------------------
    if caller_id == agent_id:
        return {"error": "An agent cannot compact itself."}
    from . import is_owner_caller
    if not is_owner_caller(caller_id) and target.parent_id != caller_id:
        return {
            "error": (
                f"Agent '{caller_id}' can only compact its direct children; "
                f"'{agent_id}' is not one."
            )
        }

    pmgr = getattr(runtime, "providers", None)
    active = getattr(pmgr, "_active_providers", None) if pmgr else None
    provider = active.get(agent_id) if isinstance(active, dict) else None

    is_running = runtime._running_count.get(agent_id, 0) > 0

    # --- Isolated-worker running path (§15 worker parity) -------------------
    # Under ATN_WORKER_ISOLATION the running agent's provider lives IN THE
    # WORKER process, so it is NOT in _active_providers (provider is None above).
    # Forward the request_compaction over the worker's IPC command channel; the
    # worker sets the flag on its live provider and reports whether an
    # orchestration was active to consume it. Only bridge/composite providers
    # stay in-process, so a supervised worker is always a generic loop (which
    # honors the flag). If the worker reports the loop wasn't active (race:
    # just ended), fall through to the idle-store path below.
    worker = await _compact_running_worker(runtime, agent_id, caller_id)
    if worker is not None:
        return worker

    # --- Running path -------------------------------------------------------
    if is_running and provider is not None:
        # A running BridgeProvider drives its loop inside the SDK subprocess;
        # the Claude Agent SDK input protocol exposes no compact control
        # message (only interrupt / setModel / setPermissionMode). Honestly
        # report unsupported rather than pretend (§15).
        from ..providers.bridge import BridgeProvider
        from ..providers.codex_bridge import CodexBridgeProvider
        if isinstance(provider, (BridgeProvider, CodexBridgeProvider)):
            return {"status": "unsupported_while_running", "agent_id": agent_id}
        # Generic loop: request_compaction() returns True iff an orchestration
        # is actively consuming (steering queue open). If the loop just ended
        # (race), fall through to the idle path.
        if hasattr(provider, "request_compaction") and \
                provider.request_compaction(requested_by=caller_id):
            return {"status": "queued", "agent_id": agent_id}

    # --- Idle path ----------------------------------------------------------
    return await _compact_idle_store(runtime, agent_id, caller_id, provider)


async def _compact_running_worker(
    runtime: Runtime, agent_id: str, caller_id: str,
) -> dict[str, Any] | None:
    """Forward a manual compaction to an isolated worker over IPC (§15).

    Returns:
      - ``{"status": "queued", ...}`` if the agent is running in a supervised
        worker and its live loop accepted the request_compaction flag;
      - ``None`` if there is no supervised worker for this agent, OR the worker
        reported no active loop to consume it (race: the run just ended) — the
        caller then falls through to the idle-store path.

    A NO-OP (returns None immediately) when worker isolation is off or no
    supervisor exists, so the flag-off path is byte-for-byte unchanged.
    """
    cfg = getattr(runtime, "_config", None)
    wi = getattr(cfg, "worker_isolation", None)
    if not getattr(wi, "enabled", False):
        return None
    supervisor = getattr(runtime, "supervisor", None)
    if supervisor is None or not supervisor.is_supervised(agent_id):
        return None
    worker = supervisor.get(agent_id)
    channel = getattr(getattr(worker, "handle", None), "channel", None)
    if channel is None:
        return None
    try:
        ack = await channel.send_cmd_await(
            "request_compaction", {"requested_by": caller_id})
    except Exception:
        log.debug("request_compaction cmd failed for %s", agent_id, exc_info=True)
        # Couldn't reach the worker's loop — fall through to the idle path.
        return None
    if isinstance(ack, dict) and ack.get("unsupported"):
        # A running bridge/SDK worker: no compact control (matches the
        # in-process bridge branch's answer exactly).
        return {"status": "unsupported_while_running", "agent_id": agent_id}
    accepted = bool((ack or {}).get("accepted")) if isinstance(ack, dict) else False
    if accepted:
        return {"status": "queued", "agent_id": agent_id}
    # Loop wasn't active (just ended) — let the caller try the idle store.
    return None


async def _compact_idle_store(
    runtime: Runtime, agent_id: str, caller_id: str, provider: Any,
) -> dict[str, Any]:
    """Summarize an idle agent's persisted conversation store in place (§15).

    Summarizes all turns but the last 2 into one summary turn (same
    Goal/Progress/Decisions/Critical-context/Next-steps template used by the
    running-loop compactor), archives the original history via the store's
    reset() archive mechanics, writes back the compacted history, then evicts
    and closes the cached provider so the next run rebuilds from the store.
    Emits CONTEXT_COMPACTION events with manual: true + requested_by.
    """
    from ..events import Event, EventType
    from ..providers.base import (
        COMPACTION_SUMMARIZER_SYSTEM,
        COMPACTION_SUMMARY_PROMPT,
    )

    store = runtime.get_agent_conversation_store(agent_id)
    turns = store.get_turns()
    turns_before = len(turns)

    _RETAIN = 2  # last N turns kept verbatim (matches _COMPACT_RETAIN_TURNS)
    _manual = {"manual": True, "requested_by": caller_id}

    async def _emit(status: str) -> None:
        await runtime.events.emit(Event(
            type=EventType.CONTEXT_COMPACTION,
            source=agent_id,
            data={
                "agent_id": agent_id,
                "pre_tokens": sum(len(t.content) for t in turns) // 4,
                "status": status,
                **_manual,
            },
        ))

    # Nothing to fold — the retained tail already IS the whole thing.
    if turns_before <= _RETAIN:
        return {
            "status": "compacted", "agent_id": agent_id,
            "turns_before": turns_before, "turns_after": turns_before,
            "note": "nothing to summarize (at or under the retained tail)",
        }

    await _emit("in_progress")

    to_summarize = turns[: turns_before - _RETAIN]
    retained = turns[turns_before - _RETAIN:]
    original_request = turns[0].content if turns else ""

    # Resolve a provider to run the summarizer: reuse the cached instance if
    # present, else build one for the agent (do NOT cache it — we evict below).
    summarizer = provider
    if summarizer is None:
        try:
            defn = runtime.get_agent(agent_id)
            summarizer = runtime.providers.resolve_provider_with_fallback(defn)
        except Exception as exc:
            await _emit("hard_truncated")
            log.warning("compact_agent: no provider for %s (%s)", agent_id, exc)
            return {"error": f"Cannot resolve a provider to summarize '{agent_id}': {exc}"}

    # Build the summarizer input in canonical message shape from the turns.
    summary_input = [
        {"role": ("assistant" if t.role == "assistant" else "user"),
         "content": (t.content if t.role in ("user", "assistant")
                     else f"[system] {t.content}")}
        for t in to_summarize
    ]
    summary_input.append({"role": "user", "content": COMPACTION_SUMMARY_PROMPT})

    summary_text = ""
    try:
        resp = await summarizer.send(
            messages=summary_input,
            system=COMPACTION_SUMMARIZER_SYSTEM,
            model="",
            max_tokens=4096,
        )
        summary_text = (resp.text or "").strip()
    except Exception as exc:
        log.warning("compact_agent: summarizer send failed for %s: %s", agent_id, exc)

    if not summary_text:
        # Never destroy history: leave the store untouched and report honestly.
        await _emit("hard_truncated")
        return {"error": f"Summarizer produced no output for '{agent_id}'; "
                         "store left unchanged."}

    summary_turn = (
        f"[Context compaction — conversation summary follows]\n\n"
        f"{summary_text}\n\n"
        f"---\n\n"
        f"[Original request]\n{original_request}\n\n"
        f"Continue from where you left off. Do not repeat completed work."
    )

    # Archive the ORIGINAL history (store.reset() moves active.jsonl -> an
    # archived <session>.jsonl and clears the in-memory buffer), then write the
    # compacted history back. reset() never destroys — it archives.
    archived_session_id = store.reset()
    store.add_user_turn(summary_turn)
    for t in retained:
        if t.role == "assistant":
            store.add_assistant_turn(t.content, execution_id=t.execution_id)
        elif t.role == "system":
            store.add_system_turn(t.content)
        else:
            store.add_user_turn(t.content)

    turns_after = store.turn_count()

    # Evict + close the cached provider so the next run rebuilds from the
    # compacted store (bridge session_id dropped). Mirrors update_agent's
    # eviction (runtime/__init__.py:912-918) minus file deletion.
    pmgr = getattr(runtime, "providers", None)
    active = getattr(pmgr, "_active_providers", None) if pmgr else None
    if isinstance(active, dict):
        old = active.pop(agent_id, None)
        if old is not None and hasattr(old, "close"):
            try:
                await old.close()
            except Exception as exc:
                log.warning("compact_agent: failed to close provider for %s: %s",
                            agent_id, exc)
        cache = getattr(pmgr, "_cached_session_stats", None)
        if isinstance(cache, dict):
            cache.pop(agent_id, None)

    await _emit("completed")

    result: dict[str, Any] = {
        "status": "compacted", "agent_id": agent_id,
        "turns_before": turns_before, "turns_after": turns_after,
    }
    if archived_session_id:
        result["archived_session_id"] = archived_session_id
    return result


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
    "probe_tools": _probe_tools,
    "register_tool": _register_tool,
    "publish_tool": _publish_tool,
    "vet_tool": _vet_tool,
    "check_evidence": _check_evidence,
    "run_trial": _run_trial,
    "adopt_tool": _adopt_tool,
    "attest_tools": _attest_tools,
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
    "get_my_budget_status": _get_my_budget_status,
    "get_usage": _get_usage,
    "metering_report": _metering_report,
    "propose_task": _propose_task,
    "list_tasks": _list_tasks,
    "get_user_profile": _get_user_profile,
    # Delegation
    "delegate_status": _delegate_status,
    "delegate_message": _delegate_message,
    "delegate_collect": _delegate_collect,
    "get_latest_thought": _get_latest_thought,
    "get_children_status": _get_children_status,
    # On-chain
    "register_on_chain": _register_on_chain,
    # Services rail (docs/services_market.md)
    "find_services": _find_services,
    "register_service": _register_service,
    "pay_for_service": _pay_for_service,
    "request_service": _request_service,
    # Manual compaction (§15)
    "compact_agent": _compact_agent,
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


# Agent-callable confinement (H2). The tools a sub-agent may invoke via
# route_tool_call/execute_tool are EXACTLY the schema-backed tools in ``_TOOLS``.
# The extra ``_EXECUTORS`` entries marked "UI-facing" above (send_agent_message,
# approve_task, reject_task, the *_conversation ops) are internal sinks reachable
# only by the trusted orchestrator or the WS/surface layer — NEVER by an agent.
# Left agent-reachable they are confused-deputy escalations: send_agent_message
# runs with surface=None and bypasses the single-writer input arbiter (spoofing
# human input); approve_task/reject_task let an agent self-approve its own
# proposed tasks. The gate in ``execute_tool`` enforces this for every agent
# caller (both the worker framework_tool RPC path and the in-process loop).
_AGENT_CALLABLE_TOOLS: frozenset[str] = frozenset(t.name for t in _TOOLS)


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


def get_all_tool_defs() -> list[dict[str, Any]]:
    """Return all core tool definitions (name, description, input_schema)."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in _TOOLS
    ]


def get_core_tool_def(name: str) -> dict[str, Any] | None:
    """Return a single core tool definition by name, or None."""
    for t in _TOOLS:
        if t.name == name:
            return {"name": t.name, "description": t.description, "input_schema": t.input_schema}
    return None


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
    # Agent-callable confinement (H2). A real sub-agent may only invoke
    # schema-backed tools; the orchestrator (caller_id None, or explicitly the
    # orchestrator id) is trusted and unrestricted. This refuses UI-facing /
    # internal executors (send_agent_message -> arbiter bypass, approve_task ->
    # self-approval, conversation ops) BEFORE the executor runs, whether the name
    # arrived via a compromised worker's framework_tool RPC or a prompt-injected
    # tool_use in the in-process loop.
    if caller_id is not None:
        from . import is_owner_caller
        if not is_owner_caller(caller_id) and name not in _AGENT_CALLABLE_TOOLS:
            log.warning("blocked non-agent-callable tool %r by caller %s",
                        name, caller_id)
            return {"error": f"Tool '{name}' is not available to agent "
                             f"'{caller_id}'"}
    # Inject caller context so tools that need it can derive parent_id
    if caller_id is not None:
        input = {**input, "_caller_id": caller_id}
    try:
        return await executor(runtime, input)
    except Exception as exc:
        log.exception("Tool execution error: %s", name)
        return {"error": f"Tool '{name}' failed: {exc}"}
