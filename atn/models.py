"""Core data models for the agent framework.

These are plain dataclasses — no ORM, no Pydantic (yet).
Keep them serialisable to JSON for persistence and wire transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentType(Enum):
    GENERAL = "general"
    EXPLORE = "explore"
    IMPLEMENT = "implement"
    RESEARCH = "research"
    DEBUG = "debug"
    REVIEW = "review"
    SPONSOR = "sponsor"


class StepType(Enum):
    SCRIPT = "script"           # deterministic shell/script execution
    COGNITIVE = "cognitive"     # LLM reasoning (future)
    MESSAGE = "message"         # post to another agent's inbox
    PULL = "pull"               # read another agent's output store
    COLLECT = "collect"         # await result of prior async message step


class AgentMode(Enum):
    PIPELINE = "pipeline"       # deterministic step sequence
    COGNITIVE = "cognitive"     # autonomous LLM session


class AgentStatus(Enum):
    REGISTERED = "registered"   # defined but not active
    ACTIVE = "active"           # armed — scheduler + inbox triggers honoured
    RUNNING = "running"         # at least one execution in progress
    STOPPED = "stopped"         # manually deactivated
    ERROR = "error"             # last execution failed
    COMPLETED = "completed"     # one-shot agent finished (parent decides removal)
    BUDGET_PAUSED = "budget_paused"  # budget exceeded, paused until budget increased


class ExecutionStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class MessagePriority(Enum):
    LOW = "low"                 # pick up whenever
    NORMAL = "normal"           # process on next run
    HIGH = "high"               # wake up if idle
    URGENT = "urgent"           # wake up immediately


class MessageType(Enum):
    TRIGGER = "trigger"         # "run now"
    WORK = "work"               # data / task to process
    INFO = "info"               # informational, no action required
    ALERT = "alert"             # something needs attention


class OnboardingStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TaskStatus(Enum):
    PROPOSED = "proposed"       # orchestrator suggested it
    APPROVED = "approved"       # user approved it
    ACTIVE = "active"           # being worked on (agent running or calendar event created)
    COMPLETED = "completed"
    REJECTED = "rejected"       # user rejected the proposal


class TaskType(Enum):
    AUTOMATION = "automation"   # create/run an agent
    CALENDAR = "calendar"       # block time on user's calendar
    REMINDER = "reminder"       # remind the user to do something


# ---------------------------------------------------------------------------
# Agent identity & economic sovereignty
# ---------------------------------------------------------------------------

@dataclass
class AgentIdentity:
    """Cryptographic identity for a registered agent."""
    public_key: str = ""                    # Hex-encoded public key
    address: str = ""                       # Derived 0x address
    lineage_hash: str = ""                  # hash(parent_hash + charter + public_key)
    parent_lineage_hash: str | None = None  # Parent's lineage hash
    registered_on_chain: bool = False
    registration_tx: str | None = None
    contract_address: str | None = None     # Deployed agent contract


@dataclass
class AlignmentVector:
    """Semantic embedding of agent's charter for alignment checking."""
    embedding: list[float] = field(default_factory=list)
    source_hash: str = ""                   # SHA256 of the text that was embedded
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    constitution_similarity: float = 0.0
    sponsor_similarity: float = 0.0


@dataclass
class AgentWallet:
    """Economic identity for a registered agent."""
    address: str = ""
    balance: dict[str, int] = field(default_factory=dict)  # token_address -> amount
    total_earnings: int = 0
    total_spent: int = 0
    sponsor_funded: bool = False


# ---------------------------------------------------------------------------
# Step & Agent definitions
# ---------------------------------------------------------------------------

@dataclass
class StepDefinition:
    """One step in an agent's execution pipeline."""
    type: StepType
    config: dict[str, Any]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.type.value


@dataclass
class HeartbeatConfig:
    """Intrinsic heartbeat — Runtime pings parent on interval while agent is active."""
    interval: str = "5m"                    # ping frequency: "30s", "5m", "1h"
    on_complete: str = "notify_parent"      # "notify_parent" | "self_deactivate"


@dataclass
class AgentDefinition:
    """Blueprint for an agent — pipeline or cognitive, with hierarchy and heartbeat."""
    id: str
    name: str

    # --- Execution mode ---
    mode: AgentMode = AgentMode.PIPELINE
    # PIPELINE: execute steps[] sequentially (original behavior)
    # COGNITIVE: autonomous LLM session (replaces delegates)

    # --- Pipeline mode fields (ignored in COGNITIVE mode) ---
    steps: list[StepDefinition] = field(default_factory=list)

    # --- Cognitive mode fields (ignored in PIPELINE mode) ---
    provider: str | list[str] = ""          # provider name or fallback chain
    cognitive_model: str = ""               # model override (e.g. "claude-opus-4-6")
    system_prompt: str = ""                 # inline text or file reference
    task_prompt: str = ""                   # the initial work assignment / prompt
    agent_type: str = "general"             # explore, implement, research, debug, review
    max_turns: int = 50                     # per-session turn limit
    tools: list[str] = field(default_factory=list)
    # Tool surface: "sdk_builtin" (file/bash/web from bridge),
    #               "atn_core" (create_agent/post_message/get_snapshot/get_output),
    #               "atn_full" (all orchestrator tools),
    #               "connectors" (MCP connectors)

    # --- Common fields ---
    concurrency: int = 1                    # 1 = singleton
    schedule: str | None = None             # e.g. "30s", "5m", "1h"
    output_schema: dict | None = None       # expected shape of final output
    description: str = ""
    budgets: dict[str, int] = field(default_factory=dict)
    # Token budgets per provider.  e.g. {"gemini": 50000, "claude_max": 100000}
    # Limit is total tokens (input+output) per execution.  0 or absent = unlimited.
    connector_ids: list[str] = field(default_factory=list)
    # MCP connectors this agent uses.  Matched against ConnectorManager specs.

    # --- Tool exposure ---
    expose_as_tool: bool = False            # if True, pipeline is callable as a tool
    tool_input_schema: dict[str, Any] | None = None
    # Custom input schema for when this agent is exposed as a tool.
    # If None and expose_as_tool=True, derived from the first step's expected input.

    # --- Hierarchy ---
    parent_id: str | None = None            # None = root (orchestrator) or user-created
    created_by: str = ""                    # agent_id of creator
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Parent notification ---
    notify_parent: bool = True              # if False, skip inbox post on completion
    # When True (default), framework posts a lean completion notification
    # to the parent's inbox.  When False, child must use post_message
    # explicitly if it has something to report.  Failures always notify.

    # --- Heartbeat ---
    heartbeat: HeartbeatConfig | None = None
    # If set, Runtime sends periodic status pings to parent_id.
    # On completion, sends a completion signal to parent.

    # --- Agent sovereign identity (None = unregistered ephemeral agent) ---
    identity: AgentIdentity | None = None
    alignment: AlignmentVector | None = None
    wallet: AgentWallet | None = None
    # Sponsor relationship
    sponsor_agent_id: str | None = None       # ID of sponsor agent funding this agent
    training_charter: str = ""                # Charter defining allowed work scope

    @property
    def model(self) -> str | None:
        """Derive the model name for display purposes.

        For cognitive mode, returns cognitive_model directly.
        For pipeline mode, derives from cognitive steps.
        """
        if self.mode == AgentMode.COGNITIVE:
            return self.cognitive_model or None
        models: list[str] = []
        for s in self.steps:
            if s.type == StepType.COGNITIVE:
                m = s.config.get("model", "")
                if m and m not in models:
                    models.append(m)
        if not models:
            return None
        return ", ".join(models)

    @property
    def is_cognitive(self) -> bool:
        return self.mode == AgentMode.COGNITIVE

    @property
    def is_pipeline(self) -> bool:
        return self.mode == AgentMode.PIPELINE

    @staticmethod
    def generate_id() -> str:
        return uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Inbox messages
# ---------------------------------------------------------------------------

@dataclass
class InboxMessage:
    """A message in an agent's inbox."""
    id: str
    source: str                 # who sent this
    target: str                 # agent_id
    type: MessageType
    priority: MessagePriority
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def generate_id() -> str:
        return uuid4().hex[:10]


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Cumulative token usage for one provider within one execution."""
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Execution records
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single step execution."""
    step_index: int
    step_name: str
    step_type: StepType
    status: ExecutionStatus
    output: Any = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    pid: int | None = None                  # subprocess PID (script steps)


@dataclass
class ExecutionRecord:
    """Full record of one pipeline execution."""
    execution_id: str
    agent_id: str
    status: ExecutionStatus
    trigger_source: str                     # "scheduler", "user", "agent:<id>", etc.
    step_results: list[StepResult] = field(default_factory=list)
    current_step: int = 0
    output: Any = None                      # final pipeline output
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    token_usage: dict[str, TokenUsage] = field(default_factory=dict)
    # Per-provider cumulative token usage for this execution (own + children).

    @staticmethod
    def generate_id() -> str:
        return uuid4().hex[:12]


# ---------------------------------------------------------------------------
# User profile & goal-oriented planning
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """User profile with goals, projects, and onboarding state.

    Forward-compatible with web3: jurisdiction_id field ready for DAO membership.
    """
    id: str = "local"                                   # user-scoped ID
    onboarding_status: OnboardingStatus = OnboardingStatus.NOT_STARTED
    summary: str = ""                                    # 2-3 sentence description
    standards: list[dict[str, Any]] = field(default_factory=list)
    # [{title, description}]
    goals: list[dict[str, Any]] = field(default_factory=list)
    # [{id, title, timeframe, description, motivation, status}]
    projects: list[dict[str, Any]] = field(default_factory=list)
    # [{id, title, description, goal_link, next_steps, status}]
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    jurisdiction_id: str | None = None                   # V0: None. Future: DAO ID
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CreditBudget:
    """Per-provider token budget for a billing period.

    The planning loop checks utilization and proactively allocates unused
    budget to goal-aligned tasks.
    """
    provider: str
    period: str = "monthly"                              # "daily" | "weekly" | "monthly"
    token_limit: int = 0                                 # 0 = unlimited
    tokens_used: int = 0
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    auto_allocate: bool = True                           # planning loop can spend unused budget


@dataclass
class PlanningTask:
    """A task proposed or managed by the planning loop.

    Links a user's goal to a concrete action: an agent automation, a calendar
    event, or a reminder.
    """
    id: str
    goal_id: str                                         # which goal this serves
    title: str
    description: str = ""
    task_type: TaskType = TaskType.AUTOMATION
    status: TaskStatus = TaskStatus.PROPOSED
    agent_id: str | None = None                          # linked agent (automation tasks)
    calendar_event_id: str | None = None                 # linked gcal event (calendar tasks)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @staticmethod
    def generate_id() -> str:
        return uuid4().hex[:10]
